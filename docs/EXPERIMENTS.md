# AgentRec-X — Experiments

Methodology and accepted results, written so a reader can reproduce or dispute them.
Every number is sourced; nothing here is estimated.

Cross-references: [README](../README.md) · [Architecture](ARCHITECTURE.md) ·
[Usage](USAGE.md).

**Result categories are kept separate throughout.** Category **A** is the recommendation
benchmark, **B** is offline policy diagnostics, **C** is engineering/system measurement.
They are never merged into one "model performance" table, because they answer different
questions with different kinds of evidence.

---

## 1. Research Questions

**RQ1.** Can a sequential recommender be exposed as a trustworthy agent tool without
letting conversational state corrupt behavioural history?

**RQ2.** Can candidate-scoped product metadata grounding enrich recommendations without
changing the SASRec candidate universe?

**RQ3.** Can explicit user preferences be represented as persistent, auditable state that is
distinct from interaction history?

**RQ4.** Can deterministic preference evidence alter candidate order while preserving the
original recommender scores and the candidate universe?

**RQ5.** Can the full pipeline operate reproducibly inside a multi-turn web agent?

**Explicitly out of scope:** *does preference reranking improve user satisfaction?* There
are no preference-conditioned relevance labels and no user study in this repository, so that
question is not answered and no proxy for it is reported.

---

## 2. Dataset

Amazon Reviews 2023, **Sports & Outdoors**, canonical identity `parent_asin`.

| Property | Value |
| --- | --- |
| Interaction file | `Sports_and_Outdoors.jsonl.gz`, 2,634,864,204 bytes |
| Raw interaction SHA-256 | `8f6ddb51c2674d048387533f1c58e22840e5eee639a59737eae979d66a87d642` |
| Metadata file | `meta_Sports_and_Outdoors.jsonl.gz`, 1,037,418,105 bytes |
| Raw metadata SHA-256 | `f75abcf0af21db0a6c6701f29d1f9b94480557ecbdc238dfc39ebf83a3c9fc3a` |
| Review source | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz` |
| Metadata source | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_Outdoors.jsonl.gz` |

`parent_asin` is used because it groups colour/size/style variants of one product and
matches the parent-level item ids used by earlier Amazon benchmarks. It is treated as an
**opaque string** — nothing in the codebase assumes a `B` prefix.

Raw data is never committed and never modified in place.

### 2.1 Engineering 100k prefix — not a benchmark

During integration, a bounded **100,000-record prefix** of the raw file was processed with
relaxed thresholds (`min_user_interactions = 2`, `min_item_interactions = 2`) into separate
`Sports_and_Outdoors_sample_*` artifacts. It exists to prove the pipeline runs end to end on
real data.

**It is not a benchmark.** A prefix of a gzipped file is not a sample of the 5-core
benchmark population, so its sparsity, counts and any metrics derived from it say nothing
about the difficulty of the final benchmark. No number from it is reported as a result in
this repository, and it is never used to infer final benchmark difficulty.

---

## 3. Preprocessing

Config: iterative k-core with `min_user_interactions = 5`, `min_item_interactions = 5`,
`pad_id = 0`, `first_real_id = 1`, `deduplicate_policy = 'last'`, `max_k_core_rounds = 100`,
ordering *timestamp ascending per user; ties by (parent_asin, rating)*.

| Stage | Value |
| --- | --- |
| Total records read | 19,595,170 |
| Parse errors | 0 |
| Duplicate records dropped | 215,135 |
| Accepted records | 19,380,035 |
| k-core rounds | 11 (converged) |
| **After filtering:** interactions | **3,500,587** |
| **After filtering:** users | **412,445** |
| **After filtering:** items | **156,746** |
| Sequence length (mean / median / min / max) | 8.4874 / 6.0 / 5 / 696 |
| Density after filtering | 5.4147e-05 |
| Retained fraction (interactions / users / items) | 0.180629 / 0.039923 / 0.098755 |

The raw input digest recorded in the preprocessing metadata matches the digest recorded in
the accepted run manifest, which ties the processed artifacts and the benchmark to the same
raw file.

**Invariants later layers depend on:** chronological ordering is preserved; item id 0 is
PAD and never a real item; ids are deterministic and re-derivable; filtering is a true
iterative fixed point rather than a single users-then-items pass.

---

## 4. Temporal Evaluation Protocol

Frozen as `agentrecx.eval_protocol.v1`, `cohort.protocol = temporal_leave_two_out`.

For a chronological sequence `[i1, ..., i(n-2), i(n-1), in]` with `n >= 3`:

```text
train history    = [i1, ..., i(n-2)]
validation       : history = train history              target = i(n-1)
test             : history = train history + [i(n-1)]   target = in
```

* **Validation history is exactly the train history.** The test history additionally
  contains the validation interaction, because that interaction happened strictly before
  the test target. Ordering — not item uniqueness — is what prevents leakage.
* **Eligibility requires `len(sequence) >= 3`**, applied at the evaluation layer.
  Preprocessing thresholds are never adjusted to make more users eligible. Exclusion is
  reported (`SplitReport`) rather than hidden. On the accepted cohort, 412,445 of 412,445
  users were eligible and **0 were excluded**.
* **Full-catalogue ranking.** No sampled negatives in the canonical benchmark; there is no
  sampled-negative result in this repository.
* **PAD (item id 0) is never a candidate** — excluded positionally from the score matrix.
* **Already-consumed items are masked** out of the candidate set.
* **The target remains eligible even if it repeats in the history** — it is removed from the
  masked set before masking is applied, so a repeated target is never masked away.
* **Deterministic tie rule:** higher score ranks first; on equal score, **lower item id
  ranks first**. This is a total order, so results are independent of sort stability,
  dictionary order or platform.
* **Metrics:** HR / Recall / NDCG at k = 5, 10, 20.
* Because there is exactly **one positive per case, HR@K == Recall@K** by construction. The
  manifest asserts this explicitly (`checks."HR == Recall @5/@10/@20" = true`). This is a
  protocol property, not an implementation shortcut.
* **Checkpoint selection uses validation only.**
* **The test set is sealed until the final evaluation**:
  `test_sealed.test_used_for_selection = false`, with the recorded statement that the test
  set was not used for model selection, early stopping, hyperparameter selection or restart
  decisions.

Target rank is computed as *the number of eligible candidates that beat the target, plus
one*, which is `O(num_items)` and provably equal to an explicit full sort. Both
implementations are tested against each other, including a randomised tie-heavy comparison.

---

## 5. SASRec Model

Architecture configuration (accepted manifest, `model_config`):

| Parameter | Value |
| --- | --- |
| Model | SASRec |
| `num_items` | 156,746 |
| `hidden_size` | 64 |
| `num_blocks` | 2 |
| `num_heads` | 2 |
| `dropout` | 0.2 |
| Activation | `gelu` |
| Normalization | pre-norm |
| Feed-forward multiplier | 4.0 |
| `initializer_range` | 0.02 |
| `layer_norm_eps` | 1e-8 |
| `max_seq_len` | 50 |
| `padding_idx` | 0 |

**`max_seq_len` selection.** Candidate windows `[20, 50, 100, 200]` were scored by raw
transition retention, with a target of ≥95 %, and the **smallest** candidate meeting it was
chosen:

| Window | Retained transitions | Retention |
| --- | --- | --- |
| 20 | 2,089,299 / 2,263,252 | 0.92314 |
| **50** | **2,223,283 / 2,263,252** | **0.98234** |
| 100 | 2,252,521 / 2,263,252 | 0.995259 |
| 200 | 2,260,873 / 2,263,252 | 0.998949 |

The same `max_seq_len = 50` is used at inference, so serving matches training.

---

## 6. Training Setup

| Parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0 |
| Batch size | 256 |
| Max grad norm | 5.0 |
| Seed | 2026 |
| Precision | fp32 |
| Device | `cuda:0` |
| Outer epoch budget | `MAX_EPOCHS = 200` |
| Early-stopping patience | `PATIENCE = 10` |

Note on the manifest's `optimizer_config.epochs = 1`: that is the trainer's own single-epoch
configuration. The canonical training loop drives one epoch per iteration and applies the
outer budget and patience itself, which is why the manifest records
`training.epochs_completed = 17` alongside `epochs = 1`.

Negative sampling is **deterministic and epoch-aware** and `resample_negatives = False` in
the canonical configuration, so the sampling schedule is a function of `(seed, epoch)`
rather than of process state.

Additional recorded checks: parameter finiteness, PAD-embedding non-drift, and non-finite
loss/gradient detection that aborts the run rather than continuing silently.

---

## 7. Canonical Full-ranking Evaluation

| Property | Value |
| --- | --- |
| Cohort | 412,445 validation cases and 412,445 test cases |
| Catalog | 156,746 items |
| Evaluation batch size | 2,048 |
| Validation runtime | 44.91 s |
| Test runtime | 45.32 s (9,100.11 users/s) |
| Protocol version | `agentrecx.eval_protocol.v1` |

The evaluator — not the model — owns masking and ranking. Models emit one score per
catalogue item and never mask, rank or compute metrics, which is what keeps the comparison
protocol identical across models.

---

## 8. Accepted SASRec Results

**Category A — recommendation benchmark.** Source: `runs/sasrec_canonical_2026/run.json`,
`run_id = m5b-sasrec-canonical-seed2026`, seed 2026, checkpoint SHA-256
`352bd3ae…a105912`.

**Validation** (used for checkpoint selection):

| Metric | @5 | @10 | @20 |
| --- | --- | --- | --- |
| HR | 0.0092351707500394 | 0.01531113239341003 | 0.02385530191904375 |
| Recall | 0.0092351707500394 | 0.01531113239341003 | 0.02385530191904375 |
| NDCG | 0.00595726128233579 | 0.00791319748084027 | 0.010062270127073334 |

**Test** (sealed, opened once after selection):

| Metric | @5 | @10 | @20 |
| --- | --- | --- | --- |
| HR | 0.008304137521366486 | 0.013565445089648317 | 0.02148892579616676 |
| Recall | 0.008304137521366486 | 0.013565445089648317 | 0.02148892579616676 |
| NDCG | 0.0053328165345728285 | 0.007019975041142503 | 0.009008710911775646 |

**Selection and cost:**

| Item | Value |
| --- | --- |
| Best epoch | 6 |
| Selection criterion | validation NDCG@10 = 0.00791319748084027 |
| Best validation HR@10 | 0.01531113239341003 |
| Epochs completed | 17 (early stopping, patience 10 exhausted) |
| Global steps | 27,404 |
| Train time | 1,627.6649248301983 s |
| Peak GPU memory | 4,997,753,344 bytes |
| Environment | RTX 4090, Python 3.10.8, torch 2.1.2+cu118, CUDA 11.8 |

Only the cutoffs 5/10/20 are canonical. No other cutoff is reported because none was
evaluated.

### 8.1 ItemCF — no comparable comparison

The manifest records:

```text
itemcf_comparison = "PENDING (no same-artifact full-data ItemCF benchmark exists)"
```

`recommendation/baselines/` contains ItemCF as an **engineering** baseline developed against
the small integration sample. It shares neither the artifact nor the full data with the run
above, so presenting a SASRec-vs-ItemCF table would compare incomparable protocols. **No
such comparison is made in this repository.**

---

## 9. Metadata Integration

**Category C/A boundary — this is a descriptive data statistic, not a quality metric.**

Source: `data/processed/Sports_and_Outdoors_products_manifest.json` (M8-A build), joined
against the accepted SASRec mapping.

| Property | Value |
| --- | --- |
| Raw metadata records seen | 1,587,421 |
| Parse errors / non-object / missing `parent_asin` / duplicate keys | 0 / 0 / 0 / 0 |
| Records matching the recommender catalog | 156,746 |
| Records outside the recommender catalog | 1,430,675 |
| Processed catalog-only records | 156,746 |
| Processed records outside catalog | **0** |
| Catalog items with metadata | 156,746 |
| Catalog items missing metadata | 0 |
| Coverage | 100.0000 % |
| Artifact | `Sports_and_Outdoors_products.jsonl`, 307,500,147 bytes, SHA-256 `175c83ca…1e71dd` |

**What `processed records outside catalog = 0` does and does not mean.** The artifact is
built with `catalog_only = true`, so out-of-catalog source records are *deliberately
excluded at build time*. The zero does **not** mean the Amazon metadata source contained no
out-of-catalog products — it contained 1,430,675 of them. Read correctly: **every item in
the served catalogue has a metadata record**, and the served artifact contains nothing else.

**Coverage is not field completeness.** 100 % catalogue coverage means every catalog item
has a record; it says nothing about which fields that record populates. Field population
across the covered catalog:

| Field | Records populated |
| --- | --- |
| `average_rating`, `rating_number` | 156,746 |
| `title` | 156,737 |
| `details` | 155,398 |
| `store` | 155,343 |
| `categories` | 152,893 |
| `features` | 144,766 |
| `main_category` | 141,850 |
| `price_text` | 94,490 |
| `description` | 91,196 |

So `price_text` is present for ~60 % of the catalogue and `description` for ~58 %. A
candidate lacking a field is still a valid candidate: it is never dropped, replaced, or
filled with generated text.

---

## 10. Preference-memory Validation

**Category C** — correctness of the memory lifecycle, validated by tests and the M9 smoke
rather than by a metric.

The contract distinguishes three operations:

| Operation | Trigger | Result |
| --- | --- | --- |
| **ADD** | default | new `active` entry at the next `logical_seq` |
| **REPLACE** | explicit correction marker (`instead`, `instead of X`, `make that`, `I meant`) | newer entry active; corrected entry `superseded` with `superseded_by`/`supersedes` links |
| **REMOVE** | explicit retraction (`"I don't care about color anymore"`) | matching entries tombstoned as `removed`; no new entry, no history erased |

Validated properties:

* **Coexistence** — "I don't want red" then "I don't want blue" keeps both avoidances; no
  kind is a singleton slot.
* **Replacement is never inferred** from the preference kind; it requires explicit
  correction intent.
* **Duplicate suppression** — an already-active `(kind, value, polarity)` is skipped.
* **Idempotency** — the same `(user_key, source_turn_id)` with the same candidates is a
  no-op; identity is origin-keyed, not text-keyed.
* **Provenance** — every entry keeps its source turn and the exact user span.
* **Isolation** — records are keyed by an explicit `user_key`; there is no process-global
  store.
* **No behavioural coupling** — the memory package has no representation for a
  `parent_asin` history, so no preference can become an interaction event.

Turn semantics, validated end to end at the HTTP level: a preference stated in a turn is
persisted during that turn and is **read at the start of the next turn**; it does not affect
the turn that created it.

---

## 11. Preference-evidence Evaluation

**Category B/C** — M10A produces evidence, not a metric.

Three-state outcome per (candidate, ACTIVE preference):

| Status | Meaning |
| --- | --- |
| `MATCH` | the metadata positively satisfies the preference |
| `VIOLATION` | an explicitly forbidden value is present, or a numeric bound is crossed |
| `UNKNOWN` | the available metadata cannot decide |

**The conservative asymmetry is the key interpretive fact:**

| Situation | Result | Why |
| --- | --- | --- |
| `avoid red`, metadata `Color = red` | `VIOLATION` | finding a forbidden value is proof |
| `prefer red`, metadata `Color = red` | `MATCH` | finding a wanted value is proof |
| `prefer red`, metadata `Color = blue` | `UNKNOWN` | metadata is not an exhaustive specification |
| `avoid red`, metadata `Color = blue` | `UNKNOWN` | absence of the forbidden value is not proof of absence |
| `avoid red`, no colour field | `UNKNOWN` | missing attribute |
| no metadata record | `UNKNOWN` | nothing to read |

**Positive preferences never produce a violation.** Only negative categorical constraints
and numeric bounds can.

This is why coverage is low and why that is not a bug: on the accepted cohort, **only
93/100 candidates had at least one decisive (non-`UNKNOWN`) observation**, and across
preference-kind observations **162/500 were known**. Per kind:

| Preference kind | Known observations / total | Note |
| --- | --- | --- |
| `price_max` | 93 / 100 | structured numeric field, best covered |
| `color` | 59 / 300 | `details.Color` is often absent or differently valued |
| `category` | 10 / 100 | free-text support, only partial by construction |

Reading these as "the system fails to notice metadata" would be wrong: a *readable field
holding a different value* is deliberately `UNKNOWN`, because the alternative would be to
report a negative fact the data does not support.

---

## 12. Reranking Policy Diagnostics

**Category B — policy diagnostics, not relevance metrics.** Source:
`experiments/reranking_evaluation_smoke.py` (M10C), which reranks a deterministic real
cohort through the **production** reranker and then measures the result offline.

Setup: the first 20 eligible users in accepted sequence order (100 candidates total), each
matched against a **globally fixed synthetic preference fixture** — *synthetic explicit
preference fixtures for policy diagnostics*. The fixture is declared before any candidate
output is examined, is not derived from user history or candidate metadata, and is **not**
observed real-user preference data.

| Diagnostic | Value |
| --- | --- |
| Requests with movement | 15 / 20 = 0.75 |
| Candidates moved | 58 / 100 (promoted 31, demoted 27, unchanged 42) |
| Mean absolute rank displacement | 0.94 |
| Maximum displacement | 4 |
| Top-1 overlap | 8 / 20 = 0.40 |
| Top-3 overlap | 49 / 60 = 0.816667 |
| Top-5 overlap | 100 / 100 = 1.0 |

Adherence at k (before → after), where the prefix is measured against explicit preference
evidence only:

| k | Violations | Matches | UNKNOWN |
| --- | --- | --- | --- |
| 1 | 2 → 0 | 27 → 41 | 71 → 59 |
| 3 | 7 → 1 | 94 → 102 | 199 → 197 |
| 5 | 16 → 16 | 146 → 146 | 338 → 338 |

k = 5 is unchanged because the policy reorders *within* the candidate set rather than
changing membership; the top-5 prefix over a 5-candidate pool is the whole pool.

Integrity diagnostics:

| Check | Result |
| --- | --- |
| `duplicate_original_rank_count` | 0 |
| `item_id` fallback reachable | false |
| Policy-order violations | 0 / 20 requests |
| Violation-protection inversions | 0 / 11 comparable pairs |
| Invariant failures | none |
| Movement accounting (`sum(causes) == moved_count`) | consistent (58 = 58) |
| Deterministic report digest | `c8fe88efbdc961889d3034607f7f7a44f997fc8a809a44acb3849e115985585b` |

**Movement is attributed to actual comparisons**, not to labels: promotions are attributed
to the highest overtaken candidate and demotions to the candidate directly above, so each
cause names the dimension that decided that pair (`fewer_violations`, `more_matches`,
`ordinal_fallback`). `item_id_tiebreak` is 0 everywhere, consistent with the proof that
`item_id` is unreachable when original ranks are unique.

### 12.1 What these numbers support

Reranking **increases adherence to the configured explicit-preference policy** under the
evaluated synthetic fixtures, changes 58 % of candidate positions in a 20-request cohort
with an unchanged candidate set and unchanged raw scores, and never uses a tie-break.

### 12.2 What they do not support

No relevance, satisfaction, conversion or quality claim. There are no
preference-conditioned relevance labels, so NDCG / HR / Recall / CTR / conversion /
satisfaction must not be attached to reranking. Policy adherence is not a relevance metric.

---

## 13. Real-chain Integration Validation

**Category C** — the pipeline works as a system.

| Check | Milestone | Result |
| --- | --- | --- |
| Agent drives the real Tool and accepted checkpoint | M7C | E2E suite green; 40 smoke gates |
| Candidate set unchanged by enrichment | M8 | enrichment is order- and identity-preserving |
| Preference memory cannot change candidates | M9 | asserted; memory never reranks |
| Tool called once per request; engine loaded once | M7C/M11 | verified by object identity across requests |
| M10A evidence covers only candidates × active preferences | M10A | asserted |
| M10B emits a permutation of its input | M10B | count, ids, identities, scores preserved |
| M10D wires evidence and policy into the route | M10D | 21 smoke gates, real chain |
| M11 multi-turn flow over real HTTP | M11 | 39 smoke gates, real chain |
| Cross-milestone consistency | M10D vs M11 | the same fixed preference fixture on the same profile produces the **same** reranked order through the agent and through HTTP (`moved_count = 3`) |
| Conversational claims never become history | M11 | history digest unchanged after purchase/click claims, asserted at the API level |
| Session isolation | M11 | two sessions on one profile have distinct `user_key`s and disjoint preference memory |

---

## 14. Latency / Engineering Diagnostics

**Category C — engineering diagnostics from the accepted local CPU setup** (RTX 4090 host,
CPU inference). These are environment-specific, vary run to run, and are **never** compared
against a recommendation-quality metric.

| Measurement | p50 | p95 |
| --- | --- | --- |
| M10A matcher (M10D smoke, k = 5) | ~0.46 ms | — |
| M10B reranker (M10D smoke, k = 5) | ~0.11 ms | — |
| M10A + M10B added overhead | ~0.58 ms | — |
| M10C offline diagnostics per request (20 requests) | ~0.78 ms | ~1.09 ms |
| M11 session creation (20 samples) | ~1.8 ms | ~2.0 ms |
| M11 chat, recommend route (12 samples) | ~48–55 ms | ~50–73 ms |
| M11 chat, direct route (6 samples) | ~7–9 ms | ~8–11 ms |
| Total graph latency, one request (M10D smoke, CPU) | ~55 ms | — |

Values marked `~` moved between repeated runs of the same smoke on the same machine — the
M10A matcher p50 was observed at both 0.463 ms and 0.465 ms, and the total graph latency
between 55.3 ms and 55.5 ms. The M11 figures come from a live server over loopback HTTP and
also varied between measurements (recommend-route p50 was seen at both 48.3 ms and 54.6 ms),
which is why they are given as ranges.
All of these are single-machine CPU measurements with run-to-run variance, not
hardware-independent performance claims.

Interpretation: the preference stages are ~0.6 ms against a ~55 ms request, i.e. roughly
1 % — the request cost is dominated by SASRec scoring over the full catalogue. The direct
route is ~7× cheaper than the recommend route because it performs no inference at all.
Latency is **not** part of any deterministic report payload, so it cannot perturb a digest.

---

## 15. Reproducibility

**Accepted commit and artifacts.**

| Item | Value |
| --- | --- |
| Current accepted commit (M11 web demo) | `c7c3bf5b2d764e3a8ca177de41e7d9080c3f5c0f` |
| Accepted checkpoint | `runs/sasrec_canonical_2026/best.pt`, SHA-256 `352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912` |
| Run manifest | `runs/sasrec_canonical_2026/run.json`, SHA-256 `e3549049f955bb0540444c221523444b2a333e8df5eac6b97984342e40c6e2c7` |
| Metadata artifact | `data/processed/Sports_and_Outdoors_products.jsonl`, SHA-256 `175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd` |
| Processed mappings | SHA-256 `dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912` |
| Processed sequences | SHA-256 `d3d834260c867ff39ac1b32bb08fd17ea1dc394e9a6fb7af8d19aa046c1eab81` |
| Raw interactions | SHA-256 `8f6ddb51c2674d048387533f1c58e22840e5eee639a59737eae979d66a87d642` |
| Benchmark run commit (recorded in manifest) | `859fd0bbbfe8a000f496a3ed4b67a6e229df56c9` |

The benchmark stays tied to the commit recorded inside its own manifest. A future commit
does **not** inherit these digests, and nothing in the repository implies it does.

**Determinism controls.**

* seed 2026 fixed for model init, negative sampling, epoch shuffling and restart;
* negative sampling is epoch-aware and reproducible (no process-global RNG state);
* the evaluation tie rule is a total order, so ranking is platform-independent;
* accepted run manifest records model, optimizer, protocol version, cohort definition and
  artifact digests together;
* the checkpoint is selected on validation only and the test set is opened once, after
  selection, as recorded in `test_sealed`;
* reranking is a pure function of the evidence report;
* offline report digests exclude timing and machine metadata, so repeated runs share a
  digest — the accepted M10C real-cohort digest `c8fe88efbdc96188…` was reproduced
  byte-identically by the Milestone 11 re-run;
* the smoke scripts that touch the real chain are the same code the tests use, and each
  prints explicit PASS/FAIL gates rather than relying on exit status alone.

**How to reproduce, by cost class.**

| Goal | Cost | Entry point |
| --- | --- | --- |
| Verify the accepted checkpoint serves | minutes | `experiments.sasrec_inference_smoke`, `experiments.web_demo_smoke` |
| Re-run the full test suite | minutes | `pytest -q` |
| Rebuild the metadata artifact | ~1 min for the build itself | `experiments.prepare_product_metadata` |
| Re-run policy diagnostics | minutes | `experiments.reranking_evaluation_smoke` |
| Re-train SASRec on full data | hours, GPU | `experiments.sasrec_canonical --canonical` |
| Re-open the sealed test set | minutes, after training | `experiments.sasrec_formal_test` |

Re-training writes **new** run directories. It does not overwrite
`runs/sasrec_canonical_2026/`, and the benchmark above remains tied to its recorded commit
and digests. The full command reference is in [USAGE.md](USAGE.md).

---

## 16. Threats to Validity / Limitations

**Evaluation.**

* The benchmark is **offline and full-catalogue**, with one positive per case. It measures
  next-item ranking accuracy on a held-out interaction; it says nothing about satisfaction,
  diversity, novelty or business outcomes.
* HR@K equals Recall@K by protocol construction, so a table showing both is not
  corroborating evidence.
* Cutoffs are limited to 5/10/20; no other cutoff was evaluated.
* ItemCF is **not** a valid comparison point (different artifact and data scale).

**Preference layer.**

* Preference evidence is **metadata-bound**. Items whose records lack the relevant field
  produce `UNKNOWN`, so observed adherence understates what a richer catalogue would allow.
* The conservative asymmetry deliberately refuses to infer absence, which caps achievable
  recall of real violations.
* Preferences in the diagnostics are **synthetic fixed fixtures**. They exercise the policy;
  they are not a model of any real shopper.
* There are **no preference-conditioned relevance labels**, so no statement about
  recommendation quality improvement from reranking can be supported.

**System.**

* Performance numbers are single-machine CPU measurements with run-to-run variance and are
  not hardware-independent.
* The web demo has **no authentication** and a **process-local session registry**.
* The demo's trusted histories come from a small fixed set of profiles derived from the
  accepted artifact with the leave-one-out test target excluded; they are not a user
  directory.

**Retrieval.**

* Candidate-scoped RAG is **lexical BM25**. It was not benchmarked against dense retrieval,
  and no claim is made that it improves recommendation accuracy — it supplies grounded
  evidence within an already-fixed candidate set.
