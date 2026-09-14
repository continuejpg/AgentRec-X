# Deterministic preference-aware reranking (Milestone 10B)

Reorders the candidates of an accepted M10A evidence report under an explicit,
auditable lexicographic policy.

```
RecommendationTool / SASRec          (upstream — not called here)
    ↓
original ordered candidates
    ↓
M8 metadata
    ↓
M9 ACTIVE preferences
    ↓
M10A MATCH / VIOLATION / UNKNOWN evidence
    ↓
M10B reranker                        (this package)
    ↓
same candidates, new order
```

> **M10B reorders candidates but never adds or removes candidates.**

> **M10B never modifies the raw SASRec score.**

> **UNKNOWN evidence is neutral.**

## 1. What M10B is

A pure function over already-computed evidence. It answers exactly one question:

> Given this evidence, what deterministic order should these same candidates have?

It is offline, deterministic, provider-free and framework-independent. It holds no
model, store, retriever or configuration, and it imports none of them.

## 2. What M10B is not

* it does not read `PreferenceMemoryStore` — memory lifecycle ended at M9/M10A;
* it does not parse or re-interpret preferences; it never inspects ADD / REPLACE /
  REMOVE, superseded or removed state, or source text;
* it does not re-run metadata retrieval, M10A matching, SASRec, the RecommendationTool,
  FastAPI or any LLM;
* it does not filter candidates — **even a VIOLATION only demotes**;
* it does not compute a weighted or final score.

It consumes one thing: `PreferenceEvidenceReport` from
[`../preference_matching/`](../preference_matching/README.md), and it uses evidence
**statuses** as the ranking contract. Provenance stays attached for audit only.

## 3. Canonical ordering policy

Lexicographic, with no weights and no coefficients:

```
(violation_count ASC, match_count DESC, original_rank ASC, item_id ASC)
```

| Priority | Key | Meaning |
| --- | --- | --- |
| 1 | `violation_count` ascending | **fewer explicit violations wins** |
| 2 | `match_count` descending | at equal violations, **more explicit matches wins** |
| 3 | `original_rank` ascending | at equal evidence, the **original SASRec rank** decides |
| 4 | `item_id` ascending | final deterministic fallback, guaranteeing a total order |

### Why lexicographic, and why violations come first

Raw SASRec scores are uncalibrated and preference evidence has no validated numeric
scale, so combining them would require inventing coefficients with no empirical
justification. Instead:

> M10B prioritizes explicit constraint adherence without numerically combining
> uncalibrated signals.

The policy is deliberately blunt at the top: **one fewer violation beats any number of
extra matches**, unless the violation counts are equal. This is intentional and tested:

```
A: 1 violation, 10 matches
B: 0 violations,  0 matches
        ⇒ B before A
```

The rule is human-auditable, and every output candidate carries a machine-readable
reason so a position can be explained without re-deriving it.

`item_id` is a final deterministic fallback only. Because M10A reports carry unique
positive `original_rank` values, key 3 already yields a total order for valid input;
key 4 exists so the comparator is total even for input that should never occur.

## 4. UNKNOWN is neutral

`UNKNOWN` is counted separately and enters **neither** key 1 nor key 2. It is not
rewarded and not penalised, so:

* a candidate with only UNKNOWN evidence keeps its original position relative to other
  equally-unknown candidates;
* metadata sparsity cannot change ranking;
* if every record is UNKNOWN, the output order equals the input order exactly.

## 5. Guarantees

| Guarantee | How |
| --- | --- |
| Candidate universe preserved 100 % | sorted copy; a multiset post-condition rejects any accidental add/drop/merge |
| Candidate count unchanged | one output record per input candidate |
| Identity unchanged | `item_id` and `parent_asin` copied verbatim |
| SASRec score unchanged | `sasrec_score` copied by exact float equality; never normalised, shifted or combined |
| Evidence unchanged | the same tuple of evidence records, including every UNKNOWN |
| No filtering | a violating candidate is demoted, never removed |
| Original rank retained | `original_rank` is a separate field; it is never overwritten by the new position |
| No final score | the report contains counts, ranks and reasons — no weighted number |
| Input not mutated | the M10A report, its candidates and its evidence are read-only |

## 6. Output schema

```python
RerankedCandidate(
    original_rank, reranked_rank, item_id, parent_asin, sasrec_score,
    match_count, violation_count, unknown_count,
    evidence, rerank_reason, reason_detail,
)

RerankingReport(
    candidates, candidate_count, moved_count, unchanged_count,
    sort_key, diagnostics,
)
```

`reranked_rank` is contiguous and unique from `1..N`. `original_rank` remains visible
and unchanged.

### `rank_delta`

```
rank_delta = original_rank - reranked_rank
```

* **positive → promoted** (moved earlier)
* **0 → unchanged**
* **negative → demoted**

### Rerank reasons

| Reason | Meaning |
| --- | --- |
| `fewer_violations` | placed ahead of the candidate below it because it has fewer explicit violations |
| `more_matches` | placed ahead at equal violations because it has more explicit matches |
| `preserved_original_order` | equal evidence; the original SASRec rank decided this position |
| `deterministic_tie_break` | every evidence and original-rank key tied; `item_id` decided it |
| `ranked_last` | final position; there is no candidate below it to be ordered against |

A reason describes the comparison a candidate **won** against the candidate immediately
below it. Comparing downwards is what keeps the label honest: a promoted candidate is
credited with the comparison it won rather than the one it lost against whatever now
sits above it.

## 7. Diagnostics

`RerankingDiagnostics` reports `candidate_count`, `moved_count`, `promoted_count`,
`demoted_count`, `unchanged_count`, `active_preference_count`, and per-`k`
`TopKAdherence` rows with `violations_before` / `violations_after` and
`matches_before` / `matches_after`.

These are **reranking diagnostics, not quality metrics**. They measure agreement with
explicit preference evidence only. M10B has no preference-conditioned relevance labels,
so no NDCG, HR or "recommendation quality improved" claim can be made from this
milestone. The accepted M5 SASRec benchmark remains frozen and separate.

## 8. Degenerate cases

| Input | Behaviour |
| --- | --- |
| empty report | valid empty reranking: `candidate_count = 0`, `candidates = []`; nothing fabricated |
| one candidate | stays rank 1 whatever the evidence; counts still reported, no movement |
| no active preferences | order identical to the input |
| every record UNKNOWN | order identical to the input |
| identical evidence profile | order identical to the input |

## 9. Input validation

Malformed input **fails clearly** rather than being silently repaired:

```
non-positive original_rank        → RerankingError
duplicate original_rank           → RerankingError
duplicate item_id                 → RerankingError
duplicate parent_asin             → RerankingError
```

A duplicate rank would make the order ambiguous, so it is refused rather than
renumbered.

## 10. Public API

```python
from recommendation.reranking import PreferenceReranker, rerank_candidates

report = match_candidates(candidates=enriched.items, preferences=active)   # M10A
reranked = PreferenceReranker().rerank(report)                             # M10B
# or, equivalently:
reranked = rerank_candidates(report=report)
```

`PreferenceReranker` optionally takes `diagnostic_k`; diagnostics never affect order.

## 11. Performance

The sort dominates: `O(n log n)` in the candidate count, plus linear passes for reason
attribution and diagnostics. No index, cache or model is involved. Measured on the real
chain and on synthetic candidate counts of 5 / 20 / 100 — see
`experiments/preference_reranking_smoke.py`.

## 12. Running it

```bash
# tests (offline, deterministic, synthetic fixtures)
.venv/bin/python -m pytest -q tests/test_preference_reranking.py

# smoke: fixture policy scenarios + M9 lifecycle + the real chain
.venv/bin/python -m experiments.preference_reranking_smoke
.venv/bin/python -m experiments.preference_reranking_smoke --k 5 --json /tmp/m10b.json
```

## 13. Policy evaluation (Milestone 10C)

`evaluation.py` and `evaluation_schemas.py` add an **observational** layer over this
reranker. It reuses the production `PreferenceReranker` as-is — it does not re-implement
the policy — and reports:

| Diagnostic | Definition |
| --- | --- |
| displacement | `moved_count`, promoted/demoted/unchanged, `mean`/`max`/`median abs(original_rank - reranked_rank)` |
| `top_k_overlap` | `|original_top_k INTERSECT reranked_top_k| / k` at `k ∈ {1,3,5,10}` (valid `k` only) |
| adherence | per `k`: violations / matches / UNKNOWN **before** and **after** |
| preference coverage | per active preference: candidates with non-UNKNOWN evidence (`known_count / candidate_count`) |
| candidate coverage | per candidate: `match_count`, `violation_count`, `unknown_count`, `all_unknown` |
| type coverage / movement | the same, broken out per preference kind |
| policy consistency | every adjacent pair is non-decreasing under the canonical key; expected `0` violations |
| violation protection | pairs where a violating candidate ranks above a clean one; expected `0` |
| position reasons | **ordering** distribution over *all* candidates: `fewer_violations` / `more_matches` / `ordinal_rank` / `item_id_tiebreak` / `last_position`, with `total == candidate_count` |
| movement attribution | **movement** distribution over *moved* candidates only: `fewer_violations` / `more_matches` / `ordinal_fallback` / `item_id_tiebreak`, with `attributed_total == moved_count` |
| invariants | candidate count/universe/ids/scores/evidence/rank/order/mutation all unchanged |
| baselines | restricted diagnostic orders (original, violation-only, match-only) for characterising key contributions |

### Ordering reason vs movement cause

These are two different questions with different denominators, and they are reported
separately:

* **Position reason** answers *"why does this candidate sit here, relative to the one
  below it?"* It is defined for every candidate (moved or not) and its denominator is
  `candidate_count`. It must never be presented as movement attribution.
* **Movement cause** answers *"why did this candidate's rank change?"* It is defined
  **only** for candidates where `original_rank != reranked_rank`, its denominator is
  `moved_count`, and every moved candidate receives exactly one cause, so
  `attributed_total == moved_count` always holds.

A movement cause is decided by the single comparison that produced the new position: for
a promotion, the highest-ranked candidate it overtook; for a demotion, the candidate
immediately above it. The classification is **symmetric** — it names the canonical
dimension that decided that pair from either side — so a candidate that overtook a
cleaner rival and the rival it displaced both report `fewer_violations`.

`ordinal_fallback` means the pair's evidence tied and the original SASRec rank decided
it, i.e. the ordering changed because preference evidence demoted another candidate, not
because this pair's evidence differed.

### `item_id` is unreachable for valid input

The canonical key's fourth component is consulted only when a pair shares an evidence
profile **and** an `original_rank`. The accepted M10B reranker rejects duplicate
`original_rank` values, so no ordering can ever depend on `item_id`: both the position
and movement `item_id_tiebreak` counts must be **0**. This is audited on every run via
`duplicate_original_rank_count` and `item_id_fallback_reachable`, and the field is kept
purely as a defensive diagnostic for noncanonical input.

**These are policy and adherence diagnostics, not quality metrics.** There are no
preference-conditioned relevance labels in this project, so M10C reports no HR, Recall,
NDCG, MRR, CTR, satisfaction or relevance figure — such a number would be fabricated.
`matches`/`violations` measure agreement with *explicit preference evidence*;
`top_k_overlap` measures *stability/displacement*; movement is not improvement.

Aggregates always carry an explicit numerator and denominator. Timing lives outside the
deterministic payload, so repeated runs produce the same report digest.

The restricted baselines exist only to show what each canonical key contributes. They are
never production alternatives.

## 14. Boundary summary

| Milestone | Scope |
| --- | --- |
| M10A | **Evidence**: MATCH / VIOLATION / UNKNOWN per candidate and active preference |
| **M10B** | **Order only**: deterministic lexicographic reranking of the same candidates |
| M10C | Out of scope here: critique, constraint explanation, LLM judging, hard filtering, weighted score tuning |

M10B changes no candidate identity, no SASRec score and no benchmark metric. It exposes
evidence and order; it does not claim a product is "best for you", and it does not
assert relevance, satisfaction or improvement over the frozen benchmark.
