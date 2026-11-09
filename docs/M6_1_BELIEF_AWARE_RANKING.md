# M6.1 — Belief-Aware Ranking Foundation

> **Status: IMPLEMENTED (offline lab only).** M6.1 joins a frozen trusted candidate
> snapshot, the M5 `for_ranker` projection and the accepted M10A/M10B explicit evidence
> into typed features, ranks them with three preregistered policies (A0/A1/A2) and
> evaluates them under the frozen protocol
> [`M6_1_RANKING_PROTOCOL.md`](M6_1_RANKING_PROTOCOL.md). Serving V1, RRF, the accepted
> reranker, SASRec/Two-Tower/BM25 and the M9 memory schema are **not modified**. No
> learned ranker is trained and no recommendation-quality improvement is claimed.

---

## 1. Existing code reused (nothing replaced)

| Component | Reused for |
| --- | --- |
| `recommendation.state.state_projector.for_ranker` (`RankerStateProjection`) | the only preference input to features; already typed, scoped, lifecycle-attributed, ordinal strength |
| `recommendation.state.shopping_adapter.candidate_pool_snapshot` / `ranking_snapshot` | the exact M5 digest algorithm used to verify candidate-pool and ranking references (no new digest scheme) |
| `recommendation.preference_matching.PreferenceCandidateMatcher` (M10A) | real explicit-preference evidence for A1; also reused for non-directional inferred beliefs |
| `recommendation.reranking.PreferenceReranker` / `rerank_candidates` (M10B) | A1's ordering, unmodified; A2's first two sort keys are the same explicit keys |
| `recommendation.catalog.ProductMetadata` / `MetadataIndex` / `normalize_product_record` | trusted catalogue join; absent fields stay absent |
| `recommendation.control.grounded_reasoning.parse_price` / `parse_weight_kg` | lazy, trusted numeric parsing for directional inferred attributes |
| `recommendation.evaluation.metrics.hit_at_k` / `ndcg_at_k` | optional labeled quality metrics |
| `recommendation.evaluation.experiment.paired_bootstrap_delta` (or a deterministic equivalent) | paired confidence intervals when labels exist |
| `recommendation.tools.schemas.ToolRecommendation`, `recommendation.rag.schemas.EnrichedRecommendation` | the accepted candidate/evidence types the matcher already consumes |

The M5 decision state is read only through its public projection: M6.1 does not query the
event store, does not write events and does not mutate M9 memory.

---

## 2. Candidate identity boundary

A candidate is `(parent_asin, item_id, original_rank)` copied from the trusted pipeline.

```text
trusted retrieval/fusion (SASRec, Two-Tower, BM25 -> RRF)   [not touched]
        |
        v
TrustedCandidateFeatureResolver
  - identity map validation (parent_asin <-> item_id), PAD=0 rejected
  - duplicate candidates / duplicate item ids rejected
  - contiguous original ranks preserved
  - catalogue metadata attached via MetadataIndex; missing record is explicit
  - snapshot refs/counts/digests verified against M5 snapshot events
  - ONE shared eligibility projection from the trusted evaluator
        |
        v
CandidateSnapshot   (immutable; same set for every arm)
        |
        +--> BeliefAwareFeatureBuilder --> CandidateFeatureSet
        |
        +--> policies A0 / A1 / A2 (reorder only)
```

Rules:

* A digest is **verification** material. It is never used to reconstruct candidate rows or
  feature values; the evaluation pipeline supplies the immutable snapshot in memory.
* A snapshot that does not belong to the expected user/task/turn, is stale (different turn
  key), has a different count/digest, or names a foreign identity is rejected rather than
  joined.
* Catalogue metadata is joined by the trusted `MetadataIndex`; a missing record yields
  `metadata_status="missing"` and no invented attributes.
* No candidate is ever added, removed, duplicated or replaced by the resolver, the feature
  builder or any ranking arm.

---

## 3. Feature schema

`CandidateRankingFeatures` (see `recommendation/ranking/schemas.py`) is frozen and contains
only copied or grounded values:

| Field | Meaning |
| --- | --- |
| `candidate_id`, `item_id`, `original_rank` | trusted identity/order |
| `source_ranks` | per-source rank + optional source score/score-kind, **provenance only** |
| `rrf_score`, `rrf_score_kind` | fused-score provenance, never a numeric ranking input |
| `explicit_matches` / `explicit_violations` / `explicit_unknown` | counts from the real M10A evidence |
| `explicit_evidence` | the accepted `PreferenceEvidence` records, unchanged |
| `inferred_matches` / `inferred_violations` / `inferred_unknown` | counts from soft belief evidence |
| `inferred_ordinal_support` / `inferred_ordinal_penalty` | set-relative ordinal counts for directional soft beliefs |
| `inferred_evidence` | per-belief evidence with reason codes and ordinal counts |
| `preference_strength_metadata` | belief id, kind, value, polarity, direction, scope, lifecycle, ordinal strength rank, soft flag (no probability) |
| `eligibility`, `eligibility_detail` | the shared trusted feasibility decision |
| `catalogue_fields.present/missing` | explicit attribute availability/missingness |

`CandidateFeatureSet` also carries `CatalogueCoverage`: metadata found/missing counts,
readable-field counts, supported/unsupported inferred features, and deterministic notes.
Repeated builds of the same snapshot/projection are byte-identical (canonical JSON digest).

---

## 4. Missing-evidence policy

1. A missing catalogue record or field is represented explicitly (`metadata_status`,
   `catalogue_fields.missing`); it is never imputed from a title, another field or a
   product name.
2. `UNKNOWN` preference evidence is neutral: it is neither a match nor a violation and
   contributes `0` support and `0` penalty.
3. A positive preference whose value is simply absent is `UNKNOWN`, never a violation
   (M10A semantics, reused unchanged).
4. Unsupported inferred attributes (no grounded catalogue field) are reported under
   `unsupported_inferred_features` and contribute no ranking signal; the task is recorded
   in coverage, not silently dropped.
5. A directional numeric belief (e.g. weight minimize) is grounded only from an existing
   numeric field (`price_text`, a details field containing "weight"); unparseable values
   are `UNKNOWN`.

---

## 5. Soft-preference ranking semantics

All arms start from the **same** snapshot and the **same** eligibility projection.

* **A0** — original/trusted order (`original_rank` ASC). Baseline.
* **A1** — the accepted M10B reranker key over the real M10A explicit evidence:
  `(explicit_violations ASC, -explicit_matches ASC, original_rank ASC, item_id ASC)`.
* **A2** — A1's explicit priority followed by the inferred ordinal tie-break:
  `(explicit_violations ASC, -explicit_matches ASC, inferred_ordinal_penalty ASC,
   -inferred_ordinal_support ASC, original_rank ASC, item_id ASC)`.

A2 properties, enforced structurally and by tests:

* the first two keys are identical to M10B, so **inferred evidence can never override
  explicit-preference priority**;
* inferred evidence only changes the order of candidates that tie on explicit evidence;
* `UNKNOWN` inferred evidence contributes `0/0`, so missing catalogue data never penalises
  a candidate;
* the ordinal counts are comparisons **inside the frozen candidate set** (how many other
  known-value candidates are better/worse for the direction); they are not probabilities,
  not calibrated confidence and not cross-model scores;
* no arm reads a raw SASRec logit, Two-Tower cosine, BM25 score or RRF score numerically;
  the final tie-breakers are `original_rank` then `item_id`, so ordering is total and
  deterministic;
* A2 is a conservative deterministic baseline, **not a learned fine ranker**.

Retraction/correction semantics are inherited from M5: a retracted soft belief is absent
from `for_ranker`, so A2 has no inferred signal and degenerates to A1; an explicit
correction supersedes the old belief in the state, so the corrected belief is the one the
feature builder sees.

---

## 6. Offline evaluation protocol and metric interpretation

The experiment is preregistered in
[`M6_1_RANKING_PROTOCOL.md`](M6_1_RANKING_PROTOCOL.md): fixed task set, one frozen
candidate snapshot per task, fixed eligibility projection, three arms, Top-K ∈ {1,3,5},
fixed seed, contract metrics always reported and quality metrics only when documented
labels exist.

Interpretation limits, restated because they matter:

* **Ranking change is not quality.** Change rate, top-K overlap, ordering determinism and
  coverage are contract/behavioural diagnostics, not effectiveness evidence.
* Historical purchase relevance answers "was the next purchase retrieved", not "did the
  user prefer the inferred feature". M6.1 therefore reports
  `NOT_EXECUTED_NO_LABELS` for quality on the synthetic tasks by default.
* Synthetic preference scenarios validate ranking *behaviour* (candidate preservation,
  explicit priority, inferred tie-break direction, UNKNOWN neutrality) but cannot prove
  user satisfaction or Team Accuracy. Those belong to a future CoShop-compatible decision
  evaluation.
* No LLM is involved, and no statistical significance is claimed without labels and
  adequate sample size.

---

## 7. Known feature-coverage gaps (executed synthetic fixture)

The preregistered synthetic tasks use 7 hiking-backpack candidates × 4 state variants
(default / retracted / corrected / directional) = 28 frozen candidate observations.

Executed coverage (aggregate over the four tasks):

| Field | Readable candidates / 28 |
| --- | --- |
| `title` | 24 |
| `price` | 20 |
| `weight` (a details key containing "weight") | 20 |
| `color` | 24 |
| `material` | 20 |
| `brand` (`store` or brand detail) | 24 |
| `features` | 24 |
| `categories` | 24 |
| metadata record present | 24 (4 explicit `missing`) |

Findings, reported rather than hidden:

* **Price/weight are only partly covered.** One candidate has no metadata record and one
  has a placeholder price token plus no weight detail; these are explicit
  `metadata_status="missing"` / `catalogue_fields.missing`, never imputed.
* **Supported inferred features:** text `feature:lightweight` (real M10A partial-token
  semantics) and directional `feature:weight (minimize)` (grounded numeric parsing).
* **Unsupported inferred feature:** `feature:durability (minimize)` — no grounded catalogue
  field, so it contributes no ranking signal and is listed under
  `unsupported_inferred_features`.
* **Explicit `comfort`** works only because the synthetic catalogue actually contains a
  comfort-bearing field; on a real catalogue without such text the same preference would
  be `UNKNOWN` and would not penalise candidates.
* **Directional inferred beliefs** are composed onto the real `for_ranker` projection only
  in the fixture's `directional` variant, because the current M5 event vocabulary
  (`PREFERENCE_HYPOTHESIS_PROPOSED`) has no `direction` field. This is an M5/M6 boundary
  gap for M6.2 to decide, not a fabricated production signal.
* The hard budget constraint is intentionally **not** a `for_ranker` feature; it reaches
  ranking only through the shared trusted eligibility projection, exactly as the protocol
  requires.
* Catalogue readability checks in `features.py` are local structural checks (so the
  feature module stays import-light); directional evidence uses the accepted
  `parse_price`/`parse_weight_kg` parsers, and a test asserts the two agree on the fixture.

---

## 8. Verification and executed findings

Executed on 2026-10-30 with the Python 3.10 test venv (the repository `.venv` is a Linux
venv). Commands:

```bash
python -m pytest -q tests/test_m6_1_resolver.py tests/test_m6_1_features.py \
    tests/test_m6_1_policies.py tests/test_m6_1_evaluation.py tests/test_m6_1_contract.py
python -m pytest -q tests/test_m6_1_resolver.py tests/test_m6_1_features.py \
    tests/test_m6_1_contract.py          # no-torch light subset
python -m experiments.m6_1_ranking_smoke
python -m experiments.m6_1_belief_aware_evaluation
```

### 8.1 Executed test results

| Check | Result |
| --- | --- |
| Full M6.1 gate (resolver + features + policies + evaluation + contract) | **113 passed** |
| Torch-free light subset (resolver/features/contract; model-stack-dependent tests skip) | **47 passed, 28 skipped** |
| `experiments/m6_1_ranking_smoke` | **SMOKE: PASS, 77/77 checks**, 4 variants, exit 0 |
| M5.2 regression subset (unchanged) | **83 passed** |
| Existing M5/shopping/docs regression batch | **192 passed** |
| Production-boundary contract test (no serving/agent/demo module imports `recommendation.ranking`) | passed |

### 8.2 Frozen paired evaluation (contract diagnostics)

The paired evaluation runner exits 0 with:

* protocol `m6.1.paired-ranking.v1`, protocol digest
  `de2e08c44f55f3bff224475a7d3b36285c8bf481c0f0d47d1463b58989398193`;
  report digest `1de4a3ea3983fd719a2440630008352b4bc1080a66e31f720c43dbe92cda7d30`,
  identical across reruns;
* 4 tasks, 28 candidate observations, **12 rankable** after the single shared eligibility
  projection (16 INELIGIBLE/UNRESOLVED excluded and never rankable in any arm);
* every arm: identity preserved, candidate set equals the **expected** pool, ordering
  deterministic, eligibility invariant violations = **0**.  The literal
  `candidate_set_equal_to_snapshot` column is `False` only because the single trusted
  eligibility projection removes INELIGIBLE/UNRESOLVED candidates (16 of 28 observations
  here); the expected-pool column is `True` for every arm and cross-arm candidate-set
  equality is asserted by the evaluator;
* A0 and A1 do not reorder relative to each other on this fixture
  (`change_vs_A0 = change_vs_A1 = 0`), while A2 changes 2/12 positions
  (`change_vs_A1 = 0.1667`) exclusively inside explicit-evidence ties;
* explicit evidence totals in every arm: 0 violations / 26 matches;
* inferred-evidence coverage: 11 candidates with decided inferred evidence, 10 with only
  UNKNOWN evidence (the UNKNOWN candidates contribute 0/0 and are not penalised);
* Top-K overlap vs A0/A1 stays 1.0 at K=1/3/5 on this small fixture (the single reorder is
  outside the top-1 both before and after; it is reported as a contract diagnostic, not a
  quality result);
* **quality status: `NOT_EXECUTED_NO_LABELS`** — the fixture has no documented
  preference-conditioned relevance labels, so **no quality metric, no paired quality delta
  and no effectiveness claim** is made.

The smoke table for the default variant (eligible candidates only; ineligible/unresolved
rows are shown with `None` ranks and their eligibility status) demonstrates A0 order, A1
explicit order, inferred match status and UNKNOWN neutrality; the directional variant
demonstrates an actual A2-vs-A1 tie-break (candidate #5, lighter under weight-minimize,
moves ahead of #2 while both tie on explicit evidence); the retracted variant makes A2
order exactly equal A1, proving the inferred signal disappears.

### 8.3 Serving V1 safety

`recommendation/ranking/` is imported only by tests and experiments. The contract test
scans the production packages for imports of the offline package (none), and the existing
shopping/agent/M5 regression suites are re-run unchanged (192 + 83 passed above). No
production env flag, default, ranking path, RRF behaviour, reranker, M9 memory or model
checkpoint is modified. Default Serving V1 remains byte-identical in behaviour.


---

## 9. Readiness for a trained fine ranker (M6.2)

M6.1 provides what M6.2 needs and nothing it does not:

* one frozen, typed feature schema with explicit missingness and ordinal-only belief
  signals, so a future model sees the same inputs as the deterministic baselines;
* a frozen candidate set and a single shared eligibility projection, so paired training /
  evaluation cannot change membership between arms;
* stable candidate identity and provenance (`parent_asin`/`item_id`, source ranks, M5
  candidate/ranking snapshot digests) for joining presentation/feedback logs;
* the explicit-priority invariant and A2's ordinal tie-break as strong deterministic
  baselines a learned ranker must beat under the same protocol.

**Preconditions before M6.2 training:** the M6.1 contract invariants must pass with zero
violations; preference-conditioned relevance labels (not purchase-only labels) must exist
under a frozen protocol; and the learned model must be evaluated paired against A0/A1/A2
with the same frozen protocol and no serving change. M6.1 itself makes no quality claim.
