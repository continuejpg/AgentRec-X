# M4 PREREGISTRATION — Fixed vs LLM-adaptive Source Selection under Matched Candidate Capability

**Status: FROZEN.** Frozen on 2026-09-28. No M4 run may start, and no M4 result may be inspected,
until every artifact hash in §12 has been verified against this document.

This document is a preregistration, not a report. It exists so that the question below is decided
by a protocol fixed in advance, and so that a result — whichever way it goes — cannot be
re-interpreted, re-tuned or quietly re-run.

**M4 is not a repair, re-run or reinterpretation of M3.** M3 remains frozen historical evidence
and is neither modified nor re-executed. M4 uses a different source universe, a different depth,
a different tool budget and a different question.

---

## 1. Research question

> When both arms have the **same four recommendation sources**, the **same per-source candidate
> depth (100)**, the **same `CandidateLedger` RRF**, the **same evaluator** and the **same tool
> budget**, does **LLM-adaptive source-subset selection with early stopping** improve
> recommendation quality, and what does it cost?

The scientific variable is therefore exactly:

> **fixed all-source retrieval  vs  LLM adaptive source-subset selection / early stopping**

### 1.1 What M4 explicitly does not answer

Written here so that no later result can be read as answering them:

* whether fusion beats a single source — **both arms fuse**;
* whether adaptive **query construction** has value — query/seed construction is frozen (§5);
* the value of `TIGER`, `popularity` or Phase-5 `metadata_retrieval` — not in the source universe (§2);
* **end-to-end live retrieval latency** — source heads are pre-materialised (§9);
* whether traversal *order* matters — M4 measures **subset selection**, and §11 records the
  invariance check that licenses that claim;
* any conclusion of M3.

---

## 2. Source universe (identical for both arms)

```text
{HISTORY, CATALOG_SEARCH, SIMILAR_ITEM, TWO_TOWER}
```

| source | registered tool | keying | availability evidence |
|---|---|---|---|
| `HISTORY` | `RecommendationTool` | identity-keyed (needs `history`) | frozen `sequential` head in §12 artifact 1 |
| `TWO_TOWER` | `TwoTowerSourceTool` | identity-keyed (needs `history`) | §12 artifact 2 |
| `CATALOG_SEARCH` | `CatalogSearchSource` | argument-keyed | `recommendation/control/catalog_search.py:143` |
| `SIMILAR_ITEM` | `SimilarItemSource` | argument-keyed | `recommendation/control/similar_item.py:192`; index built by `build_similar_item_index` (`similar_item.py:252`) from the trusted catalogue, no missing artifact |

The identity-keyed / argument-keyed split is `recommendation/control/candidate_plane.py:393-401`.

**Excluded, with reasons:**

| excluded | reason |
|---|---|
| `TIGER` | not wired into the canonical runtime; `tiger.pt`, the SID artifacts and `retrieval_candidates.jsonl` are absent on this machine, so no consistent head can be produced |
| `popularity`, `metadata_retrieval` (Phase-5 arms) | **not `CandidateSource` members** — the agent cannot reach them structurally. Including them would create a new asymmetry instead of removing one |

**Wiring requirement.** Both arms register `similar_item_tool`. By the frozen rule at
`recommendation/control/loop.py:521-535`, `SELECT_SOURCE` is offered **only** when
`CandidateSource.SIMILAR_ITEM` is registered. Without it, source selection is not a real choice.
This wiring changes **no** line of `loop.py`.

---

## 3. Depth

Per source, **100**.

Hard upper bounds, all verified:

| location | bound |
|---|---|
| `candidate_plane.py:185` (history) | `max(1, min(limit, 100))` |
| `recommendation/control/two_tower_source.py:98` | `max(1, min(int(limit), 100))` |
| `recommendation/control/arguments.py:150,173,298` | `le=MAX_K` = **100** |

Heads shorter than 100 are **truncated as produced — never zero-padded, never topped up from
another source**. Per-source, per-user realised lengths are written into the materialised artifact
and their distribution is reported.

---

## 4. Fusion

**No fusion action is added.** The final ranking is read deterministically from
`CandidateLedger.ranked_parent_asins()` (`recommendation/control/candidate_ledger.py:304`), which
fuses via `reciprocal_rank_fusion` (`:171`) with `RRF_K = 60` (`:64`) — **the same implementation
`experiments/benchmark_public.py:61` imports**. It is not reimplemented.

Two properties were verified before freezing and are relied upon:

1. **Single-source equivalence.** `1/(k + rank)` strictly decreases in `rank`, so with one source
   the fused order is exactly that source's own order. Verified for all four sources
   (`['A','B','C','D'] → ['A','B','C','D']`, etc., `identical=True` in every case).
2. **Order invariance.** See §11.

---

## 5. Query and seed construction (frozen, one implementation for both arms)

```text
SEARCH_CATALOG.terms          := deterministic_terms(h)
FIND_SIMILAR.seed_parent_asin := newest_indexed(h)
```

Both are pure functions of `h` alone: no randomness, no dependence on catalogue-wide statistics,
no dependence on the ledger, the model, or the wall clock. The implementation file and this rule
text are hash-bound (§12 artifact 6).

### 5.1 Definition of `h` (exact, no post-result freedom)

`h` is the run's **trusted test-time history**, as parent ASIN identities, in chronological order:

```text
h := tuple(
        identity.item_id_to_parent_asin(int(i))
        for i in case.test_history
        if 0 < int(i) < len(id2item) and id2item[int(i)] is not None
     )
```

Bound to concrete repository fields, in this order:

| step | source |
|---|---|
| `case.test_history` | `recommendation/evaluation/split.py:95-104` — the property returning `train_history + (validation_target,)` |
| `case.train_history` | `recommendation/evaluation/split.py:74-75` — `i1..i(n-2)` |
| `case.validation_target` | `recommendation/evaluation/split.py:76-77` — `i(n-1)` |
| identity mapping and the validity filter | `experiments/m3_agent_harness.py:274-279` (the harness convention M4 reuses) |
| what the source tools actually receive | `recommendation/control/loop.py:829,864` → `read_trusted_history` → `candidate_plane.py:398` |

**`validation_target` IS part of `h`.** Under `temporal_leave_two_out` the validation interaction
happened strictly before the test interaction, so it is legitimately visible when predicting
`test_target`. This matches the evaluator, which masks with `case.test_history`
(`recommendation/evaluation/split.py:106-114`, `test_seen`).

**The model never sees `h`.** `build_policy_context_payload`
(`recommendation/control/model_policy.py:242-300`) discloses only `has_trusted_history`, a boolean,
and its docstring states trusted behavioural history is deliberately absent. Query *content* is
therefore supplied by the system, never by the provider; the provider selects only *which source*
to consult. This is what makes §1's exclusion of "adaptive query construction" true by
construction rather than by promise.

**Cross-arm identity.** Both arms call the same two functions on the same `h` read through the same
seam, so for a given user the `CATALOG_SEARCH` and `SIMILAR_ITEM` candidate lists are byte-identical
across arms. A test asserts this directly.

---

## 6. Budget and action space

```text
max_tool_calls = 4      # exactly four source retrievals
max_steps      = 6      # four sources + FINISH + headroom
```

Budget accounting is `loop.py:1128-1132`:
`consumed_tool_call = validated.action in TOOL_CALL_ACTIONS`, where
`TOOL_CALL_ACTIONS = CANDIDATE_ACTIONS + EVIDENCE_ACTIONS`
(`recommendation/control/schemas.py:256-259`).

| action | consumes budget | evidence |
|---|---|---|
| `SELECT_SOURCE` | **yes** | member of `CANDIDATE_ACTIONS` |
| `FINISH` | **no** | member of `NON_EXECUTING_ACTIONS` (`schemas.py:262-266`) |
| read-only reasoning actions | no | M4 configures no reasoning executor, so they are never offered |

`FINISH` and the ranking read-out therefore cost **zero** source calls.

---

## 7. The two policies

### 7.1 Fixed arm (no LLM)

```text
for source in (HISTORY, CATALOG_SEARCH, SIMILAR_ITEM, TWO_TOWER):   # frozen order: CandidateSource.value ascending
    SELECT_SOURCE{source, limit=100}
FINISH
final_ranking = ledger.ranked_parent_asins()
```

Deterministic policy. `provider_calls = 0`.

### 7.2 Adaptive arm (LLM)

```text
per user:
  loop: menu = available_actions()              # frozen rule, unmodified
        LLM chooses SELECT_SOURCE{an unqueried source} or FINISH
        post-parse guard (§8) → ActionValidator (unmodified) → execute
        → GroundingVerifier → ledger
  until FINISH or budget exhausted
final_ranking = ledger.ranked_parent_asins()
```

Same provider, model, temperature and sampling as recorded in §12; no prompt tuning after results.
One small-sample preflight runs before the formal execution, does not touch the cohort, and is not
part of the result. Its configuration is frozen with the rest.

### 7.3 Provider-visible action space

The adaptive action space **shown to the provider** is restricted to:

```text
SELECT_SOURCE   (over the four frozen sources)
FINISH
```

`SELECT_SOURCE` is an **existing** `ActionKind`; M4 adds no new action. What M4 does is **restrict
the provider-visible adaptive action space** to those two members.

**How, without modifying `loop.py`.** Verified facts:

* `model_policy.py:515` computes `action_schema = build_action_schema(available)` **once**, uses it
  in the `ModelRequest` at `:524`, and parses against the same `available` at `:547`;
* `ModelRequest` is `frozen=True` (`model_client.py:76`) but a **replacement instance may be
  constructed**;
* `LLMAgentPolicy` accepts an **injected `model`** (`model_policy.py:381-388`) and calls it at
  exactly one site: `self._model.complete(request)`.

M4 therefore supplies an **injected schema-projection client** that, inside `complete(request)`,
constructs a replacement `ModelRequest` whose `action_schema` is filtered to
`{SELECT_SOURCE, FINISH}` and forwards that to the real provider.

**Semantic placement (must be stated this way).** This is a **policy-level projection**, not a
change to the control-plane menu. The controller still offers what the frozen rule says it offers;
the policy narrows the choice set it is willing to show the provider.

**Not relied upon, but true.** A model that somehow proposed `SEARCH_CATALOG` or
`ASK_CLARIFICATION` would have to supply required arguments (`terms`, a question) that the
projected schema never shows it, and `ActionProposal.model_post_init`
(`recommendation/control/schemas.py:346-381`) would refuse the proposal as `invalid_arguments`.

---

## 8. Hard post-response protocol guard

Provider-visible projection (§7.3) is **not sufficient on its own**, because the downstream parser
still knows the original available-action set (`model_policy.py:547`). M4 therefore adds a
**mechanical post-parse guard**, applied to every parsed proposal before anything else happens.

**C4-1. Allowed actions.** After parsing, only two actions may proceed:

```text
SELECT_SOURCE
FINISH
```

Any other parsed action is:

1. recorded as `protocol_deviation` (with the action name, step index and parse outcome);
2. **not executed**;
3. **never written to `CandidateLedger`**;
4. terminates the run according to the frozen deviation rule of §10.

**C4-2. `SELECT_SOURCE` sub-guards.** A parsed `SELECT_SOURCE` is likewise rejected — under the
same four consequences above — if either:

* `source` is **outside the frozen four-source universe** of §2; or
* that source **has already been queried** for this user.

**C4-3. Placement.** The guard lives in M4's own policy wrapper, so it changes no frozen file. It
does not pre-empt the repository's own `ActionValidator`, which continues to run afterwards as
defence in depth.

**C4-4. Test.** A test asserts, for each of these inputs, the four consequences of C4-1:
a proposal naming `SEARCH_CATALOG`; one naming `FIND_SIMILAR`; one naming
`RECOMMEND_FROM_HISTORY`; one naming `ASK_CLARIFICATION`; a `SELECT_SOURCE` naming a source outside
the universe; and a `SELECT_SOURCE` naming an already-queried source.

---

## 9. Termination rules

* **Allowed:** `FINISH` after at least one queried source. Verified equivalent to returning that
  source's own ranking, through the same ledger semantics (§4 property 1).
* **Prohibited:** `FINISH` with zero sources queried. Enforced in M4's policy. A violation is
  recorded as `zero_source_finish`, the user **stays in the primary analysis**, and an empty
  ranking is scored as a miss.
* **Prohibited:** querying the same source twice for one user (§8 C4-2). Rationale, measured before
  freezing: `record_from_source` **appends** provenance for an existing entry
  (`candidate_ledger.py:424`) and `ranked_entries` pushes every sighting of a source into that
  source's list (`:289-293`), so a repeated query **double-counts** the entry in RRF. Measured
  effect: identity `A`'s fused score rose from `0.0164` to `0.0323`, overtaking the previously
  first-ranked `B`; a single-source order changed from `['A','B']` to `['A','D','B']`.
  Three mechanisms enforce the prohibition:
  1. **entry tightening** — the policy proposes only unqueried sources. The policy tracks queried
     sources itself: `PolicyContext` does not carry them, but `CandidateSetObservation.source`
     (`schemas.py:649`) is visible per step and a policy holds cross-step state;
  2. **hard isolation** — M4's frozen source adapters **refuse** a second call for the same
     `(user, source)`, so a duplicate result can **never reach `CandidateLedger`**;
  3. **frozen termination** — on detecting a duplicate the run terminates immediately (equivalent
     to `FINISH`) and the user's **then-current** ledger ranking is evaluated.
* Budget exhaustion stops the run (`check_tool_budget`, `loop.py:772-800`); the actual output is
  evaluated.

---

## 10. Deviation handling — no user is excluded

**Primary analysis covers all 20,000 users.** A user with a recorded `protocol_deviation` is **not**
excluded. That user is evaluated on the ranking that actually resulted, following the frozen rule:
the ledger ranking at the moment of termination; if that is empty, the user is a miss.

Deviations are reported as a count and a per-user list.

**A per-protocol analysis excluding those users may be reported only as a secondary sensitivity
analysis**, clearly labelled, and it may never replace or override the primary result.

---

## 11. Source-order invariance (verified before freezing)

**Claim.** Fusion depends on *which* sources were queried, not on *the order* they were queried in.
Without this, M4 would measure traversal order as well as subset selection.

**Evidence.** With identical frozen source heads and identical source subsets, every execution
permutation was run against `CandidateLedger` and `ranked_parent_asins()` compared:

| subset | permutations | distinct outputs |
|---|---|---|
| all four sources | 24 | **1** |
| `HISTORY` + `CATALOG_SEARCH` | 2 | **1** |
| `CATALOG_SEARCH` + `TWO_TOWER` | 2 | **1** |
| `HISTORY` + `SIMILAR_ITEM` + `TWO_TOWER` | 6 | **1** |

**Result: invariant.** The mechanism is that `reciprocal_rank_fusion` sums
`1/(k + rank)` per identity, a commutative accumulation, and breaks ties by identity string, so the
result is a total order independent of source order. This check is re-run as a regression test
(§13 D8).

---

## 12. Frozen inputs and hashes

Every hash below was computed from the working tree at freeze time and must be re-verified before
the run. Artifacts 1–5 are inputs; artifact 6 is the query-rule implementation once written; the
M4 harness, policies and benchmark are added after freezing and their hashes recorded in the run
provenance.

| # | path | sha256 |
|---|---|---|
| 1 | `runs/m3_evidence/phase5_fusion_source_heads.npz` | `c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4` |
| 2 | `runs/twotower_public_2026_full/checkpoint.pt` | `0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab` |
| 3 | `data/processed/Sports_and_Outdoors_products.jsonl` | `175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd` |
| 4 | `data/processed/Sports_and_Outdoors_sequences.json` | `d3d834260c867ff39ac1b32bb08fd17ea1dc394e9a6fb7af8d19aa046c1eab81` |
| 5 | `data/processed/Sports_and_Outdoors_mappings.json` | `dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912` |
| 6 | `experiments/m4_query_rules.py` | recorded at implementation time, before execution |

### 12.1 Frozen source files (must not change)

| file | sha256 | note |
|---|---|---|
| `recommendation/control/loop.py` | `1a509df86d3cac87e40e9d9c51e2bd3cfcd2cd3a3aeb4a92c62966bda9a44a8b` | offered-action rule read, never modified |
| `recommendation/control/candidate_ledger.py` | `fcc65e74d0efb78d3078956e89b9b3fda2cd6281d508f9b6cba02b1be579aa62` | **matches the `docs/PROJECT_STATE.md` manifest** |
| `recommendation/control/model_policy.py` | `e6c4c795c7257342eb693e4b69b640db6feb468b37078be9f816bd64488290f2` | projection is injected, not patched |
| `recommendation/control/similar_item.py` | `0d5f2b3ce250e7e7203a7a9ea4bf62924a9fcb8a3bb0653837774e230f999eed` | |
| `recommendation/control/catalog_search.py` | `b38f0903f09653d9b172bc4bf10bc7cc4e35fa29b0ed2208fad2111f68dbfd51` | |
| `recommendation/evaluation/batched.py` | `1721ae7bf60ef3dfb43f8d0f96b49cb3562865ac4eca9e178d8bb7b277832f11` | **matches the `docs/PROJECT_STATE.md` manifest** |
| `recommendation/evaluation/split.py` | `054211a685c8653237fa7a6602832de07607bf2345930cffa25fcea9af20ef08` | **matches the `docs/PROJECT_STATE.md` manifest** |

### 12.2 Frozen configuration

| item | value |
|---|---|
| cohort | the frozen 20,000-user cohort, seed `20260201`, 412,445 eligible users |
| protocol | `agentrecx.eval_protocol.v1`, `temporal_leave_two_out` |
| catalogue | 156,746 items, PAD 0 excluded |
| K | `{5, 10, 20}` |
| evaluator | `recommendation.evaluation.batched.evaluate_batched` |
| depth | 100 per source |
| fusion | `reciprocal_rank_fusion`, `RRF_K = 60`, equal weights |
| `max_tool_calls` / `max_steps` | 4 / 6 |
| provider / model / sampling | DeepSeek `deepseek-flash`, thinking disabled, temperature 0.0, JSON mode, `max_tokens` 512 — identical to the frozen M3 amendment, recorded with its hash |
| prompt | the M4 prompt, hashed before execution, never edited afterwards |

### 12.3 Implementation constraints measured before freezing

Recorded so that they cannot become post-hoc excuses, and so the materialisation pass is designed
correctly. Measured on this machine (7 GiB RAM, ~6.9 GiB available):

| measurement | value |
|---|---|
| `MetadataIndex.load(products.jsonl)` | 21.8 s, peak RSS 1.15–1.35 GiB |
| `CatalogSearchSource` build | 14.8 s, cumulative peak RSS **2.98 GiB** |
| `build_similar_item_index` | 18.3 s, **vocabulary 255,461**, peak RSS **5.93 GiB** |
| all three in **one** process | **OOM-killed** (exit 137) |
| `SimilarItemIndex.neighbours(seed, limit=100)` | **all sampled seeds returned exactly 100** (min 100, mean 100.0) |
| that query's latency | **~2.43 s** per seed → ~13.5 h for 20,000 users |
| query determinism | identical output on repeat |

**Consequences, frozen as requirements:**

* the materialisation pass (§13 D2) **must build `CatalogSearchSource` and `SimilarItemIndex` in
  separate processes** and persist between them; building all three together is out of memory;
* the pass must record **peak RSS and wall time per stage** in its provenance;
* materialised heads are therefore not an optimisation but a **prerequisite**, which is also why §9
  forbids claiming end-to-end live retrieval latency.

---

## 13. Cost semantics

Because the source heads are **pre-materialised**, the evaluator performs **no live recommender
retrieval**. What is measured is **control-plane / trajectory latency**: decision-making,
validation, ledger bookkeeping and provider round-trips.

Frozen wording for the report:

* report **`source-call count`** as a **retrieval-cost proxy** — each source call corresponds to one
  real retrieval's share of work;
* report **control-plane / trajectory latency**, per call and in total;
* **do not claim end-to-end live retrieval latency.** That phrase may not appear as a measured
  quantity. §12.3's figures are the only live-retrieval numbers and are labelled
  **reference** measurements, not experimental results;
* additionally record one **reference** wall-clock measurement of a live four-source retrieval for a
  small non-cohort sample, labelled as a reference quantity and excluded from all M4 metrics.

### 13.1 Cost ledger (reported separately, never merged with quality)

Per-user source-call count and its distribution; provider calls; input / output / total tokens;
control-plane trajectory latency (per call and total); wall time; failure count and failure codes;
retry count; empty-ranking count; `zero_source_finish` count; `protocol_deviation` count.

---

## 14. Metrics

**Primary endpoint:** `ΔNDCG@10`, paired, adaptive − fixed.

**Secondary:** NDCG@5 / @20; Recall@5 / @10 / @20; HR@5 / @10 / @20; per-user hits; **per-source
hits**; the distribution over the 16 possible queried-source subsets; mean number of sources queried.

For the single-positive protocol `HR@K ≡ Recall@K` by construction
(`recommendation/evaluation/metrics.py:370-373`); this is documented, not a bug.

---

## 15. Paired statistics

Inherited unchanged from M2/M3:

* **exact paired McNemar** for hit@K, no continuity correction,
  `p = min(1, 2 · Σ_{i ≤ min(b,c)} C(n,i) / 2^n)`;
* **paired user-level percentile bootstrap** for Recall/NDCG deltas: resample the **20,000 users with
  replacement**, **10,000 resamples**, **seed 20260201**, report the observed paired mean difference
  with the **2.5th / 97.5th percentiles** as a 95% CI.

**Classification rule, declared before results:**

| verdict | condition |
|---|---|
| `SUPPORTED` | at **every** K, McNemar `p < 0.05` **and** both the Recall and the NDCG CI exclude 0 |
| `DIRECTIONAL ONLY` | direction consistent but not every condition met |
| `NO DETECTABLE DIFFERENCE` | CIs include 0 with no consistent direction |
| `EVIDENCE MISSING` | the quantity could not be computed from the run's own artifacts |

The **primary analysis uses all 20,000 users** (§10).

---

## 16. Failure handling

* Provider failure → record the failure code; that user's ranking is **empty and scored as a miss**.
  Substituting another arm's ranking is **forbidden** (the principle of
  `experiments/m3_agent_arm.py:23-24`).
* Invalid action → fails under the repository's existing control-plane semantics; not masked.
* `protocol_deviation` → handled by §8 and §10; the user **stays in** the primary analysis.
* `zero_source_finish` → recorded, user stays in, scored as a miss for an empty ranking.

---

## 17. Stop rule and prohibited post-result changes

**One execution**, covering all 20,000 users in a single pass. No mid-run tuning. No configuration
change after any result has been inspected.

**Frozen and not changeable after this document is hashed:** the source universe (§2); depth (§3);
the fusion rule and its constant (§4); the query/seed construction and the definition of `h` (§5);
the budget and action space (§6); the two policies (§7); the provider-visible projection (§7.3);
the post-parse guard (§8); the termination rules (§9); the deviation rule (§10); the cost wording
(§13); the metrics (§14); the statistics, seeds and classification rule (§15); the failure rules
(§16); and the frozen hashes (§12).

**M3 remains frozen**: not repaired, not reinterpreted, not re-run.

---

## 18. Minimal implementation delta (after freezing)

All new files. **No frozen file is modified**: `loop.py`, `model_policy.py`,
`candidate_ledger.py` and the evaluator are untouched. No `ActionKind` is added.

| # | file | content |
|---|---|---|
| D1 | `experiments/m4_query_rules.py` | `deterministic_terms(h)`, `newest_indexed(h)`; hash-bound as §12 artifact 6 |
| D2 | `experiments/m4_materialize_heads.py` | materialise 4 sources × depth 100 → `runs/m4_evidence/heads.npz` + SHA-256 + per-source length distribution + per-stage peak RSS and wall time. **Must build `CatalogSearchSource` and `SimilarItemIndex` in separate processes** (§12.3) |
| D3 | `experiments/m4_frozen_sources.py` | four source adapters keyed per §2, each enforcing **one call per `(user, source)`** |
| D4 | `experiments/m4_harness.py` | `CandidatePlane` with all four sources registered (including `similar_item_tool`), `GroundingVerifier`, a fresh `CandidateLedger` per user; `LoopLimits(max_tool_calls=4, max_steps=6)` |
| D5 | `experiments/m4_schema_projection.py` | the injected provider-visible schema projection client (§7.3) |
| D6 | `experiments/m4_policies.py` | fixed policy (§7.1); adaptive policy (§7.2) including the §8 post-parse guard and §9 zero-source prohibition |
| D7 | `experiments/m4_benchmark.py` | runner + statistics (§15) + cost ledger (§13.1). Does **not** reuse `m3_agent_benchmark.py`'s `EXPECTED` map or statistics |
| D8 | `tests/test_m4_*.py` | ① single-source fusion ≡ source order; ② **provider-visible schema is exactly `{select_source, finish}`**; ③ the six `protocol_deviation` inputs of §8 C4-4 each produce the four consequences; ④ budget: `FINISH` is free, a fifth source call is refused; ⑤ the four-source universe and depth are identical across arms; ⑥ zero-source `FINISH` is refused; ⑦ §5 construction is deterministic and byte-identical across arms; ⑧ **§11 source-order invariance**, re-run as a regression test |
| D9 | `docs/reports/m4-*.txt` | the report, written only after execution |

**Execution order:** verify §12 hashes → D1–D7 → D8 all green → D2 materialisation with SHA-256
verification → non-cohort preflight → **one** formal execution → report.

**Side benefit of adding no `ActionKind` and modifying no shared file:** M3's request fingerprint is
unaffected, so the archived M3 recordings remain replayable.
