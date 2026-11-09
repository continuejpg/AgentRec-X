# M6.1 — Frozen Offline Ranking Protocol (Preregistration)

> **Status: PREREGISTERED before any M6.1 experimental run.** This document fixes the
> M6.1 offline ranking experiment *before* results are produced. Any later change is
> recorded as an explicit amendment with a reason; results must be read against the
> version they were produced under. Nothing in M6.1 changes Serving V1, RRF, the
> accepted reranker or the frozen evaluation protocol.

- **Protocol version:** `m6.1.paired-ranking.v1`
- **Preregistered:** 2026-10-30
- **Code under test:** `recommendation/ranking/` (offline lab package) and
  `experiments/m6_1_*.py`; M5/M5.1/M5.2 code is reused unchanged.
- **Result status at preregistration:** `NOT EXECUTED`.

---

## 1. Main research question

Can a `UnifiedDecisionState` `for_ranker` projection be joined to a frozen trusted
candidate set, turned into typed grounded features, and consumed by a deterministic
belief-aware ranking policy that:

1. preserves candidate identity, candidate-set membership and hard-constraint eligibility;
2. preserves the accepted M10B explicit-preference priority;
3. uses inferred hypotheses **only as an ordinal tie-break** among candidates equal under
   explicit-preference priority; and
4. produces identical results under replay,

without changing Serving V1 and **without claiming recommendation-quality improvement**
from ranking change alone?

The primary deliverable is an **engineering/contract** conclusion. A quality conclusion is
possible only if preference-conditioned relevance labels exist under a frozen protocol —
M6.1 does not manufacture them from purchase history.

---

## 2. Data source and eligible tasks

### 2.1 Preregistered source (executed in M6.1)

**Synthetic grounded ranking tasks** built from the repository's accepted offline
fixtures:

* candidate rows/metadata: `tests/agent_reranking_fixture.py` and
  `tests/preference_matching_fixture.py` (real `ProductMetadata` normalization; prices,
  weights, colour and brand are grounded source fields);
* candidate snapshot: a trusted in-memory immutable snapshot (identity, original rank,
  per-source ranks/RRF provenance, catalogue metadata, eligibility), never reconstructed
  from a digest;
* M5 beliefs: real `UnifiedDecisionState` built by applying validated M5 events and
  projected through `StateProjector.for_ranker`; no labels are fed into it;
* explicit-preference evidence: the accepted M10A
  `PreferenceCandidateMatcher` over the same candidates and the explicit beliefs;
* A1 reranking: the accepted M10B `PreferenceReranker`, unmodified.

The synthetic scenario is the task scenario from the milestone brief: hiking backpack,
hard budget ≤ $150, explicit `comfort` preference, inferred `lightweight` hypothesis,
some candidates with UNKNOWN attributes, and a retraction case.

### 2.2 Labeled real-data extension

Not executed in M6.1. The repository's accepted labeled protocol is temporal
leave-two-out on the frozen public cohort; it yields **retrieval-target relevance**, not
preference-conditioned satisfaction labels. If a future amendment supplies
preference-conditioned labels, they must follow the existing chronological split and
frozen evaluator; historical purchases alone must not be presented as evidence that a user
preferred an inferred feature.

### 2.3 Exclusions

A task is excluded when any of the following holds:

* the candidate snapshot fails ownership/staleness/count/digest verification;
* candidate identities are not unique, PAD/invalid, or the metadata record belongs to a
  different `parent_asin`;
* the `for_ranker` projection belongs to a different user/task/turn than the snapshot;
* a task label (when labels are supplied) names a candidate outside the snapshot;
* the task's catalogue evidence is insufficient to build the declared features — the task
  is excluded and counted under the feature-coverage report, not silently dropped.

---

## 3. Candidate-set construction and pairing

* One frozen `CandidateSnapshot` per task; **all arms rank exactly this set** after one
  shared eligibility projection.
* Hard-constraint eligibility is applied **once, before all arms**: when an eligibility
  assessment exists, only `VERIFIED_ELIGIBLE` candidates enter the ranking pool; if no
  assessment exists (`NOT_ASSESSED`) the snapshot is used as-is and this is recorded.
* An arm may reorder candidates; it may never add, remove, duplicate or replace one.
  Candidate-set equality across A0/A1/A2 is asserted per task.
* No arm retrieves anything. Retrieval/fusion happens before the snapshot exists.
* Ground-truth labels are stored only on the evaluation task and are never exposed through
  the feature builder or any policy input.

---

## 4. Ranking arms (preregistered)

| Arm | Definition |
| --- | --- |
| **A0** | Existing trusted candidate ordering (the frozen snapshot's original order / RRF baseline order). Identity ordering by `original_rank`. |
| **A1** | The accepted M10A evidence over explicit preferences, ordered by the **real, unmodified** M10B `PreferenceReranker` lexicographic policy `(violations ASC, matches DESC, original_rank ASC, item_id ASC)`. |
| **A2** | A1's explicit-preference priority, then a bounded inferred-preference ordinal tie-break, then `original_rank`, then `item_id`. Inferred evidence is used **only after** the two explicit keys are equal. |

A2 key (exact):

```text
(explicit_violations ASC,
 -explicit_matches ASC,
 inferred_ordinal_penalty ASC,
 -inferred_ordinal_support ASC,
 original_rank ASC,
 item_id ASC)
```

Semantics fixed before execution:

* `inferred_ordinal_penalty` / `inferred_ordinal_support` are grounded, set-relative
  ordinal counts: for a directional numeric belief (e.g. weight minimize), each candidate
  counts other candidates in the **same frozen set** with a strictly worse / better known
  value. They are not probabilities and not cross-model scores.
* Non-directional inferred beliefs reuse the accepted three-state matching semantics:
  `MATCH` only with positive grounded evidence, absence is `UNKNOWN`, never `VIOLATION`.
* `UNKNOWN` inferred evidence contributes `0`/`0`, i.e. neither rewarded nor penalised.
* No arm may read a raw SASRec logit, Two-Tower cosine, BM25 score or RRF score as a
  numeric ranking input. Source ranks and the RRF score are provenance only, recorded in
  the feature schema and never arithmetically mixed.
* A2 is **not** a learned ranker; it is a deterministic conservative baseline.

---

## 5. Metrics

### 5.1 Primary (contract/behavioural) metrics — always reported

* candidate identity preservation per arm (set equality to the snapshot);
* candidate-set equality across A0/A1/A2;
* ordering determinism (same features → same order, and replay after M5 event replay);
* eligibility invariant violations (no arm may rank a candidate the shared eligibility
  projection excluded; none may add one);
* change rate vs A0 and vs A1 (fraction of candidates whose position changed);
* top-K overlap vs A0 and vs A1 for K ∈ {1, 3, 5};
* explicit preference violations/matches totals before/after;
* known/UNKNOWN inferred-evidence coverage;
* ranking latency (wall-clock, reported for cost only; not compared statistically).

### 5.2 Quality metrics — only if documented labels exist

* MRR, Recall@K, NDCG@K for K ∈ {1, 3, 5}, using the accepted metric helpers with
  binary relevance (`relevance > 0`);
* paired user/task-level deltas of A2 vs A0 and A2 vs A1, with a fixed-seed paired
  bootstrap confidence interval when the label count justifies reporting one;
* label protocol and task count are recorded with the metrics.

**If pref-conditioned labels are absent, quality status is
`NOT_EXECUTED_NO_LABELS`.** Ranking change rate, overlap or any behavioural metric is
never reported as effectiveness evidence. Historical purchase relevance does not prove a
preference for an inferred feature.

---

## 6. Determinism, seeds, missing evidence

* Bootstrap seed: `20260201`; draws: `2000` (only used when labels exist).
* Candidate ordering is total and deterministic: lexicographic keys end in
  `original_rank ASC, item_id ASC`; no wall-clock, hash-order or random tie-breaking.
* Feature building is deterministic; repeated builds for the same task must be equal
  (canonical JSON digest).
* Missing catalogue attributes: explicit `UNKNOWN`, listed in feature coverage; never
  imputed, never converted into a violation, never used as a positive inferred match.
* Unsupported attribute beliefs (no grounded catalogue field) are reported under
  "unsupported inferred features" and contribute no ranking signal.

---

## 7. Failure categories (pre-declared)

1. **Snapshot contract failure** — ownership/staleness/count/digest/identity rejection.
2. **Feature coverage failure** — declared feature unsupported by catalogue; task excluded.
3. **Policy invariant violation** — candidate set changed, eligibility violated, explicit
   priority overridden.
4. **Determinism failure** — repeated build/rank differs.
5. **Label-quality failure** — labels absent, incomplete or not
   preference-conditioned; quality metrics not reported.

A run with any category-1/3/4 failure is a failed engineering run and cannot justify
M6.2 training.

---

## 8. Primary outcome and promotion criteria

**Primary outcome:** all A0/A1/A2 contract invariants hold on 100% of eligible synthetic
tasks; A2 differs from A1 only by demoting/promoting candidates that are equal under
explicit preference priority; A1 reproduces the unmodified M10B order; rankings replay
identically.

**Promotion to M6.2 (learned fine-ranking) requires:**

1. every M6.1 contract invariant passes with zero violations;
2. the explicit-priority preservation and candidate-set equality tests pass;
3. the feature schema is stable, with documented catalogue coverage gaps;
4. **preference-conditioned relevance labels** exist under a frozen protocol, and a
   paired evaluation shows a non-degenerate signal on the preregistered metrics;
5. no Serving V1 change, no new production flag and no frozen-protocol modification.

M6.1 alone may conclude **"M6.2 training is technically justified / not justified"** from
(1)–(3). It must not conclude that A2 improves recommendation quality.

---

## 9. Reporting rules

* Report the protocol version and digest with every result.
* Report contract metrics separately from quality metrics.
* Report exclusions, feature-coverage gaps and unsupported attributes explicitly.
* Never convert ranking-change counts into quality language.
* Every claim must cite an executed command and its output; unexecuted tests are marked
  `NOT EXECUTED`.

---

## 10. Amendments

None at preregistration. Any later change is appended here with date, reason and the
resulting protocol digest, and results are labelled with the protocol version they used.

---

## 11. Execution record (post-preregistration, no amendment)

Executed on 2026-10-30 with protocol `m6.1.paired-ranking.v1` and the synthetic trusted
fixture described in §2.1. No protocol rule was changed after execution; this section only
records what was actually run, so results cannot be detached from their preregistration.

* Full M6.1 test gate (resolver/features/policies/evaluation/contract): **113 passed**.
* `experiments.m6_1_ranking_smoke`: **PASS, 77/77 checks**, 4 variants
  (default/retracted/corrected/directional).
* `experiments.m6_1_belief_aware_evaluation`: exit 0;
  protocol digest `de2e08c44f55f3bff224475a7d3b36285c8bf481c0f0d47d1463b58989398193`,
  report digest `1de4a3ea3983fd719a2440630008352b4bc1080a66e31f720c43dbe92cda7d30`
  (stable across reruns); 4 tasks / 28 candidate observations / 12 rankable after the
  shared eligibility projection; every arm preserves identity, equals the expected pool,
  is deterministic and has zero eligibility-invariant violations; A2 changes 2/12
  positions vs A0 and A1, only inside explicit-evidence ties; **quality status
  `NOT_EXECUTED_NO_LABELS`** — no preference-conditioned labels exist, so no quality
  metric or improvement claim is made.
* M5.2 regression subset (untouched): **83 passed**; existing M5/shopping/docs regression
  batch: **192 passed**.

Any future labeled execution must use this same protocol (or append an amendment with its
own digest) and report paired, task-level deltas with the preregistered seed.
