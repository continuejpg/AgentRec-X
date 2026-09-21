# AgentRec-X — Phase 5 Handoff: Public Recommendation Benchmark

Compressed context for a fresh conversation. Source of truth: the repository, Git history, tests and
`docs/ARCHITECTURE.md`. This file states **what was measured, what it means, and what must not be
reopened**. It does not restate the Phase-4 agent measurement; where that matters it is cited and
attributed rather than re-derived.

---

## 1. The question this phase answers

> Does AgentRec-X provide measurable recommendation value on a real public dataset?

The phase deliberately does **not** add agent architecture. It builds the evidence that the
recommendation layer works on public data, because packaging the project as a finished system before
that evidence exists would be a claim without a measurement behind it.

---

## 2. Dataset and split

| | Value |
|---|---|
| Dataset | **Amazon Reviews 2023**, `meta_categories` + reviews |
| Category | `Sports_and_Outdoors` (one computationally manageable category) |
| Raw size | 1 037 418 105 bytes (metadata archive), SHA-256 in the artifact header |
| Users / items / interactions | 412 445 / 156 746 / 3 500 587 |
| Item identity | **`parent_asin`**, the accepted canonical identity |
| Padding | item id `0` is `PAD`; real ids are `1..156746` |
| Filtering | iterative k-core (users and items) — the accepted fixed-point semantics, not a single pass |
| Split | `temporal_leave_two_out`, `agentrecx.eval_protocol.v1`, min sequence length 3 |
| Split shapes | `train_history = i1..i(n-2)`, `validation_target = i(n-1)`, `test_target = in`, `test_history = train_history + (validation_target,)` |
| Eligible users | 412 445 (0 excluded) |
| Evaluated cohort | **deterministic 20 000-user stratified sample** (`COHORT_SEED = 20260201`) |
| Processed artifact hashes | sequences `d3d834260c867ff3…`, mappings `dca7815a2482ab20…` |

**Why a sample, stated plainly.** The accepted SASRec run evaluated all 412 445 users on an
RTX 4090. This host is CPU-only with 7 GB of RAM, and the metadata arms need the 307 MB catalogue
artifact plus the retrieval index in memory. The benchmark therefore evaluates a stratified sample:
users are bucketed by history length, each bucket is seeded-shuffled, and users are drawn
round-robin so no bucket is under-represented. Every arm sees the **same 20 000 users**, so the
*comparison* is controlled. Absolute numbers are a sample estimate and are never presented as the
accepted full-cohort figure.

**The sample tracks the full cohort.** The corrected sequential arm scores **Recall@10 = 0.013 50** on
the 20 000-user sample against the accepted full-cohort **0.013 57** — a 0.5% relative difference.
That agreement is the evidence that the sample is representative and that the arm is wired correctly.

Cached/processed artifacts are not committed (`data/`, `runs/` are git-ignored); the artifact is
reproducible from the raw corpus, the preprocessing configuration and the code revision.

---

## 3. Benchmark architecture

`experiments/benchmark_public.py` owns the protocol; `experiments/phase5_benchmark.py` is the
runner that writes one machine-readable artifact.

One evaluator, one split, one candidate universe for every arm:

* the **shared evaluator** is `recommendation.evaluation.batched.evaluate_batched`, the accepted
  full-catalogue evaluator. It owns PAD exclusion, seen-item masking, tie-breaking (higher score
  first, then lower item id) and ranking;
* an **arm supplies raw scores**, never a masked or filtered list. A model arm yields a
  `[batch, num_items + 1]` score matrix; a ranker arm encodes its ranking as a strictly decreasing
  score matrix. Both are then masked and ranked identically;
* `k ∈ {5, 10, 20}` and the cohort, protocol and seeds are recorded in every result.

**Two bugs were found and fixed while building this, and both are now regression-tested** (a wrong arm
produces a plausible, wrong table, so this is the phase's main correctness risk):

1. `ranking_batches` scored `scores[:, 1:] = base`, which scores ascending **item id** and ignores its
   `ranking` argument completely. The popularity arm was measuring "prefer the highest item id" and
   reported a mean target rank *below chance*. Fixed by scattering the ranking into item-id
   positions; `test_ranking_batches_scatters_ranking_into_item_positions` pins it.
2. `arm_sasrec` **right**-padded the model window. `full_catalog_scores` reads the hidden state at the
   *last valid position*, so the model was summarising trailing PAD rows. This silently cost ~20% of
   Recall@10 (0.010 85 vs 0.013 50). Fixed to left-pad, matching `datasets.sasrec.build_arrays` and
   `inference.sasrec.encode_history`;
   `test_sasrec_arm_left_pads_histories_into_the_model_window` pins it.

**Scale decision.** The evaluator holds a float32 score matrix *and* a same-shaped bool mask, so peak
memory grows ~1.2 KB per case at this catalogue width. `DEFAULT_BATCH_SIZE = 256` keeps the measured
peak at 3.9 GB; the earlier default of 4096 was OOM-killed on this host.

---

## 4. Baselines (5 arms, all on the same 20 000 users)

| Arm | What it is |
|---|---|
| `popularity` | item frequency counted **only over cohort train histories** — counting over all users would leak validation/test targets into the statistic. The floor every other arm must beat. |
| `sequential` | the **accepted SASRec checkpoint** (`runs/sasrec_canonical_2026/best.pt`, 10 135 104 params), used exactly as accepted: no retraining, no re-tuning, no candidate shortlist. |
| `metadata_retrieval` | BM25 item-to-item over normalized catalogue text, rank-fused across the user's 5 most recent history items. No training. Not personalized beyond the query text. |
| `fixed_fusion` | reciprocal rank fusion of all three sources (popularity + sequential + metadata) for **every** user, same order, same constant. Deliberately not adaptive. |
| `agent_selected` | source **selection** per user over the same fusion: cold start → popularity, short history (< 8 distinct items) → metadata, otherwise → sequential. |

`fixed_fusion` and `agent_selected` share one `fused_head` implementation, so the only difference
between them is which sources each user's run consults.

**The agent-selected arm is a deterministic stand-in, not a language model.** The Phase-4 live policy
answers "which source is worth consulting?" with a real model over the same information (see
§7 for those numbers), but a language model cannot be called once per user across 20 000 users. The
arm uses explicit rules over the same signals and reports each rule's firing rate. Treating the two as
interchangeable would overstate this arm.

---

## 5. Recommendation metrics (20 000 users, full-catalogue ranking)

| Arm | Recall@5 | **Recall@10** | Recall@20 | NDCG@10 | HR@10 | mean target rank |
|---|---|---|---|---|---|---|
| `popularity` | 0.003 70 | 0.005 25 | 0.007 85 | 0.002 69 | 0.005 25 | 51 752 |
| `metadata_retrieval` | 0.005 05 | 0.009 25 | 0.014 85 | 0.004 41 | 0.009 25 | 93 379 |
| `agent_selected` | 0.006 90 | 0.011 80 | 0.018 20 | 0.005 84 | 0.011 80 | 85 732 |
| `sequential` | 0.008 20 | 0.013 50 | 0.020 70 | 0.006 85 | 0.013 50 | 20 598 |
| **`fixed_fusion`** | **0.008 60** | **0.014 35** | **0.023 25** | **0.007 43** | **0.014 35** | 80 590 |

`HR@10 == Recall@10` for every arm. That is expected, not a bug: the protocol has a single positive
per user, so hit-rate and recall are numerically equivalent (documented in AGENTS.md §6).

**Full-cohort reference point.** The accepted SASRec run over all 412 445 users reports Recall@10 =
0.013 57, NDCG@10 = 0.007 02, HR@10 = 0.013 57. The sample's `sequential` row reproduces it.

**Read `mean target rank` carefully.** For `fixed_fusion` and `agent_selected` it is *not* comparable
with the model arms and is not a quality summary. A retrieval arm's unretrieved tail is a stated
convention (ascending item id below every retrieved item), so a target outside the fused head lands
deep in the tail. Recall@10 is unaffected — it depends only on the top of the ranking — which is why
the retrieval arms are compared on recall and NDCG, not on mean rank. The diagnostic that makes this
interpretable is `source_target_in_head`.

---

## 6. Ablation findings

### Does the Agent's source selection help?

**No — it is substantially worse than fixed fusion on every metric**, and worse than the sequential
arm it chooses for most users:

| | Recall@10 | vs popularity floor |
|---|---|---|
| `popularity` (floor) | 0.005 25 | 1.00× |
| `metadata_retrieval` only | 0.009 25 | 1.76× |
| `agent_selected` | 0.011 80 | 2.25× |
| `sequential` only | 0.013 50 | 2.57× |
| `fixed_fusion` (all sources) | 0.014 35 | 2.73× |

The sources are **complementary**: `sequential` retrieves the target in its top-1 000 head for 20.5%
of users, `metadata` for 8.2% and `popularity` for 10.7%, and no source contains every target another
contains. Fusing all three therefore beats the best single source. The selection policy throws that
complementarity away: it consults exactly one source for 99.96% of users (11 971 sequential, 8 020
metadata, 9 popularity), so it can only ever recover what that one source found.

**Therefore: multi-source fusion is the measured capability, and adaptive source selection is the
measured ceiling — not the other way round.** The Phase-4 live model's source selection was never
justified as a quality improvement, and this phase shows a threshold-based selection policy is
actively harmful on this dataset. What made fusion work is consulting everything, not choosing.

### What each source contributes

| Source | Share of users whose target is in its top-1 000 head |
|---|---|
| `sequential` (SASRec) | 20.50% |
| `popularity` | 10.73% |
| `metadata` (BM25 item-to-item) | 8.17% |

`popularity` is a weak *ranker* (Recall@10 = 0.005 25, roughly random for this catalogue) but a
non-trivial *retriever*: it puts the target in its head for 10.7% of users. That is why an RRF fusion
that includes it gains over the sequential source alone even though popularity alone is near the
floor.

### Prompt and budget knobs were not tuned

No threshold, no RRF constant (`k = 60`, the repository default), no `seeds`/`per_seed` setting and no
metric was tuned against the test split. The `agent_selected` thresholds (3 and 8 distinct items) were
chosen from the rule rationale, not from a sweep — and the honest result is that they perform worse
than fusing everything, which is reported rather than re-tuned away.

---

## 7. Agent efficiency metrics (Phase 4, cited — not re-measured here)

Recommendation quality and agent quality are measured separately. Phase 5 supplies the first; the
Phase-4 real-provider run supplies the second. Attribution matters, so these are the **Phase-4**
numbers on the **18-case fixture suite**, not Phase-5 numbers:

| Metric | Phase-4 live (DeepSeek `deepseek-flash`) | Replay (offline) |
|---|---|---|
| Cases | 18 | 18 |
| Passed | 10 / 18 | identical |
| Model calls | 77 | 77 replayed, **0 live** |
| Tokens in / out / total | 125 042 / 4 225 / 129 267 | identical |
| Model latency | 81 324 ms | 0 network |
| Executed actions | identical across live and replay | identical |
| Failures by component | 8 cases: 6 over-clarification, 2 no-progress repetition | identical |
| Failure types | **no** evidence-capability, retrieval or trust-boundary failure | identical |

Live failures were policy *selection*, not capability. The contrast with Phase 3.1 (9/14) is
**confounded** and must not be read as a clean improvement: Phase 3.1 was a fresh live run while
Phase 4 replays an archived one, and the Phase-4 harness offers a wider action menu
(`find_similar`, `select_source`).

---

## 8. Leakage and reproducibility checks

The runner refuses to write metrics when any check fails (`assert_protocol`). All checks passed:

| Check | Result |
|---|---|
| `timestamp_evidence.cases_checked` | 20 000 |
| `timestamp_evidence.history_strictly_before_target` | **20 000 / 20 000** |
| `timestamp_evidence.non_monotonic_sequences` | **0** |
| `timestamp_evidence.all_checked_cases_strictly_ordered` | true |
| `test_target_in_own_history` | 120 — **all** legal repurchases (`of_which_repurchase_of_an_earlier_item` = 120, `unexplained` = **0**) |
| `validation_target_in_own_history` | 127 (same repurchase semantics) |
| `repeat_share` | 0.006 |
| `sample_targets_subset_of_full` | true |
| `split_shapes_consistent` | true |

**Why the check is decided on timestamps.** The naive rule "the target must not appear in its own
history" flagged 2.74% of the full cohort during development. Investigating found repeat purchases
with distinct timestamps (minimum delta 790 ms, zero pairs with delta 0), all 412 445 users strictly
monotonic in `unix_ms`. The flagged cases were legitimate repurchase, not leakage. The check therefore
verifies that every history interaction happened strictly before the target, and *reports* the
repurchase count instead of hiding it.

**Reproduce:**

```bash
# full 20 000-user benchmark (~22 min, ~3.9 GB peak on this host)
.venv/bin/python experiments/phase5_benchmark.py 20000 --with-metadata
# -> runs/phase5_benchmark_public/arms.json   (git-ignored)

# smoke on a small cohort (~70 s)
.venv/bin/python experiments/phase5_benchmark.py 300 /tmp/p5.json --with-metadata
```

Every result records: benchmark version, dataset manifest with content hashes, split report, cohort
description, `k` values, batch size, leakage checks, per-arm protocol block, per-arm timings and peak
RSS.

---

## 9. Known limitations

1. **The evaluated cohort is a 20 000-user sample**, not the full 412 445. Comparisons are controlled;
   absolute values are estimates. The sequential arm's agreement with the accepted full-cohort figure
   (0.013 50 vs 0.013 57) bounds how far the sample drifts, but the other arms have no full-cohort
   reference.
2. **The agent-selected arm is a deterministic rule stand-in, not the live LLM policy.** The rules are
   auditable and their firing rates are reported; they are not a model.
3. **The cold-start rule is nearly unexercised.** It fired for 9 of 20 000 users, because this
   catalogue's k-core filtering leaves a minimum test history of 4 items. Any conclusion about
   cold-start behaviour on this split would be unsupported.
4. **The selection policy consults one source per user**, so the ablation says nothing about a policy
   that consults a *subset of two* — that variant was not measured.
5. **Retrieval is lexical, not learned.** `metadata_retrieval` is BM25 over catalogue text with no
   embedding model, so it cannot retrieve a semantically similar product sharing no vocabulary.
6. **The metadata arm's mean target rank is not a quality signal** (tail convention, §5). Only
   recall/NDCG at small `k` are comparable across arms.
7. **`fixed_fusion` uses the full three-source fusion with a fixed RRF constant**, so "fusion helps"
   is established for this fusion rule, not for fusion in general; a weighted or learned fusion was
   not measured.
8. **Absolute values are low in RecSys terms** (Recall@10 ≈ 1.4%). This is attributable to the
   dataset and protocol: full-catalogue ranking over 156 746 items with a single positive per user,
   and a k-core-filtered category in which most users have very short histories. It is not evidence of
   a strong recommender.
9. **`agent_selected` performs worse than `fixed_fusion`.** Reported as measured; no threshold was
   tuned afterward to reverse it.
10. **No production packaging** — the runner is an offline experiment, not a service.

---

## 10. Tests and verification

`tests/test_benchmark_public.py` (31 tests) covers: ranking direction and permutation validation,
left-padding regression, retrieval-index/`CatalogSearchSource` agreement, term-cap behaviour,
honest retrieval misses, fusion-not-concatenation, alignment validation, fixed-arm invariance, every
agent selection threshold, unservable-plan fallback, and the timestamp-based leakage checks including
the legal-repurchase case and the non-monotonic failure.

Verified at this revision on this host:

| Command | Result |
|---|---|
| `python -m pytest -q` | **1894 passed, 33 skipped** (109 s, no network, no credential) |
| `python -m compileall -q recommendation tests experiments` | clean |
| `python -m pytest tests/test_docs.py -q` | 69 passed |
| `python experiments/phase5_benchmark.py 20000 --with-metadata` | completed in 1377.6 s, peak 3.89 GB, leakage gate passed |
| `python experiments/phase5_benchmark.py 300 /tmp/p5.json --with-metadata` | smoke, 71.9 s |

The Phase-4 baseline before this phase was 1817 passed / 33 skipped, so Phase 5 adds 77 tests with no
regressions. The Agent trust-boundary tests are unchanged and still pass; Phase 5 added no agent
architecture.

---

## 11. Do Not Reopen

Do not revisit the evaluator's terminal semantics, the split protocol, the `FeasibleCandidateView`
rules, the run-scoped reference check, the recorder's modes or the prompt contract without a failing
test that justifies it.

Do not add:

* Planner / subgoals / reflection
* multi-agent architecture
* RL / SFT / reward model
* commerce (cart, checkout, payment, order)
* production API / Docker packaging
* a `NoProgressGuard` (repetition is 2 of 8 Phase-4 failures, both already bounded by existing budgets)
* prompt or threshold tuning aimed at a benchmark number

**Do not tune the benchmark toward a nicer table.** The selection result is negative and that is the
finding. Do not change the cohort to lift a score, do not change cases, and do not re-tune the
`agent_selected` thresholds to beat `fixed_fusion` — if that is ever revisited it must be a new,
pre-registered ablation.

**Next phase is Demo/Packaging**, and it is now justified on evidence: the recommendation layer has a
public-data measurement, a controlled five-arm comparison and passing leakage checks. The measured
facts to carry forward are that **multi-source fusion beats every single source**, that **adaptive
selection by threshold loses to fusing everything**, and that **absolute quality is low**, so the demo
must present a decision-support assistant with honest retrieval, not a claim of recommendation
superiority.
