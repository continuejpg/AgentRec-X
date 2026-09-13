# Unified Evaluation Protocol (Milestone 2A)

The single source of truth for how AgentRec-X recommendation models are compared.
It is **model independent**: no recommender is implemented here. ItemCF, SASRec and
SID-OneRec are built in later milestones and plug into the scorer boundary below.

This is a **full-ranking** protocol. Sampled-negative evaluation is deliberately
not implemented — the two produce incomparable numbers (AGENTS.md §6), so mixing
them would invalidate every comparison.

---

## 1. Module layout

| Path | Responsibility |
| --- | --- |
| `recommendation/evaluation/split.py` | temporal leave-two-out splitting, cohort selection, catalog, artifact loading |
| `recommendation/evaluation/metrics.py` | ranking kernel + tie rule, target rank, HR/Recall/NDCG, aggregation |
| `recommendation/evaluation/evaluator.py` | model-independent driver and scorer boundary |
| `tests/test_evaluation_split.py` | split/eligibility/catalog/artifact tests |
| `tests/test_evaluation_metrics.py` | ranking, masking, metrics, aggregation, driver + non-finite score tests |
| `tests/test_evaluation_integration.py` | real-artifact integration sanity check |

The first model that consumes this protocol is the ItemCF baseline in
[`../baselines/README.md`](../baselines/README.md). Models live under
`recommendation/baselines/`; the protocol stays here.

---

## 2. Temporal split contract

For a chronological user sequence `[i1, i2, ..., i(n-1), in]` with `n >= 3`:

```
train_history = [i1, ..., i(n-2)]
validation    : history = train_history            target = i(n-1)
test          : history = train_history + [i(n-1)] target = in
```

* **Validation history** is exactly the train history.
* **Test history** additionally contains the validation interaction, because that
  interaction happened strictly before the test target.
* The two targets are the last two interactions, so the history of each target
  contains only earlier events. Ordering — not item uniqueness — is what prevents
  leakage.

### Eligible-user policy

Users with `len(sequence) < 3` are excluded **at the evaluation layer**. The
preprocessing thresholds and artifacts are never adjusted to make more users
eligible. Exclusion is reported, not hidden (`SplitReport`), and a cohort where
*nobody* is eligible raises an explicit error rather than yielding `NaN`.

Sequences are never mutated: each split copies items into tuples, and cases are
emitted sorted by `(user_int_id, user_id)` so the cohort does not depend on
dictionary insertion order.

---

## 3. Item-id contract

* Item id `0` is PAD and is **never** a candidate, a target, or a history entry.
* Real item ids are contiguous `1..num_items`.
* The evaluation catalog is exactly `{1, ..., num_items}`.
* Invalid ids (PAD, negatives, `> num_items`, non-integers) are rejected with an
  exception instead of being silently skipped, so a corrupt artifact or a
  model-side indexing bug cannot quietly distort the cohort.

---

## 4. Full-ranking candidate semantics

For one case:

```
excluded_seen = set(history) - {target}
candidates    = {1, ..., num_items} - excluded_seen
```

* Every catalog item is a candidate; no negatives are sampled.
* **The target always stays eligible**, including when the same item also appears
  earlier in the history. Real users re-purchase products, and the integration
  sample contains such users; subtracting the target from the mask is what makes
  that case well defined. There is an explicit regression test for it.
* Seen items other than the target are masked out, so a model is never credited
  for recommending something the user already had.

---

## 5. Deterministic tie rule

Ranking order is frozen as:

1. higher score ranks first;
2. on equal score, **lower item id ranks first**.

This is a total order, so the ranking is unique and independent of sort stability,
dictionary order or platform.

### Target-rank algorithm

Because the protocol has exactly one positive per case, sorting the catalog is
unnecessary. The target's rank is the number of eligible candidates that beat it,
plus one:

```
rank = 1
for each candidate c != target:
    if score[c] > score[target]:                      rank += 1
    elif score[c] == score[target] and c < target:    rank += 1
```

Cost is `O(num_items)` with `O(1)` extra space. `metrics.sorted_ranking` /
`metrics.rank_of_target_via_sort` provide the explicit-sort oracle, and the tests
prove the two agree — including a randomised, tie-heavy, seeded comparison and a
hand-computed case.

---

## 6. Metrics

Cutoffs are configurable; canonical default `(5, 10, 20)`.

With 1-based target rank `r`:

```
HR@K     = 1.0 if r <= K else 0.0
Recall@K = 1.0 if r <= K else 0.0
NDCG@K   = 1 / log2(r + 1) if r <= K else 0.0
```

* `NDCG@K` at `r = 1` is exactly `1.0`, since `log2(2) == 1`.
* **HR@K and Recall@K are numerically identical** under a single-positive protocol.
  This is a property of the protocol, not a bug; both are kept because future
  protocols may have multiple positives. `compare_hr_recall(report)` reports the
  equivalence per K so a future divergence is visible.
* Aggregation is the arithmetic mean over cases, computed once after collecting all
  case results, so it is order independent. `EvaluationReport` also exposes a rank
  histogram, mean target rank and mean candidate count.
* An **empty cohort raises `EvaluationError`** rather than returning `NaN`.

---

## 7. Scorer boundary

A scorer is a callable:

```python
def score_fn(history: tuple[int, ...], target_item_id: int) -> Sequence[float]:
    """Return one score per catalog item: scores[item_id] for item_id in 1..num_items."""
```

`scores[0]` is the PAD slot. It is accepted and ignored only so that an indexing
off-by-one surfaces as an explicit length error (`num_items + 1`) instead of
silently shifting every rank by one.

**Scores must be finite.** `NaN` and `±inf` are rejected with an explicit
`EvaluationError`, because non-finite values silently corrupt ranking: every
IEEE-754 comparison against `NaN` is false, so a `NaN` target compares as "not
worse" against every candidate and is ranked **first**, while a `NaN` competitor is
skipped and never counted. `±inf` is rejected too, so a model cannot smuggle in an
"always rank first" sentinel — express confidence with large *finite* values. The
whole vector is checked including index 0, so a `NaN`-padded vector cannot slip
through. A model that cannot score a case should return an all-zero vector, not
`NaN`. (Regression tests: `test_non_finite_scores_are_rejected`,
`test_nan_target_would_otherwise_rank_first`.)

The evaluator — not the model — owns PAD exclusion, seen-item masking, target
retention, ranking and metric computation. Predictions are therefore never a ranked
id list: ranking a model's own shortlist would reintroduce per-model candidate
protocols, which is the thing this milestone exists to prevent.

Cases are scored one at a time, so a model never needs to materialise a score
matrix for the whole cohort.

---

## 8. Usage

```python
from recommendation.evaluation import (
    FullRankingEvaluator, build_cohort_from_artifacts, cohort_summary,
)

cases, split = build_cohort_from_artifacts(
    "data/processed/<Category>_sequences.json",
    "data/processed/<Category>_mappings.json",
)
print(cohort_summary(cases, split))          # counts + invariants only

evaluator = FullRankingEvaluator(num_items=split.catalog_size, k_values=(5, 10, 20))

def score_fn(history, target_item_id):       # later: ItemCF / SASRec
    ...                                      # one score per catalog item

for mode in ("validation", "test"):
    outcome = evaluator.evaluate(cases, score_fn, mode=mode)
    print(outcome.report.format())
    assert all(outcome.hr_recall_agree.values())
```

Both modes call the *same* `score_fn`; only the history/target selection differs.

### Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q recommendation tests
```

The real-artifact tests skip themselves when `data/` is absent (it is
git-ignored), so a fresh clone stays green.

---

## 9. Interpretation rule

The Milestone 1.5 sample is a 100k-record *prefix* used as an engineering
integration fixture. Cohort statistics derived from it (eligible users, case
counts, history lengths) describe this sample only and must **not** be read as
properties of the full Sports and Outdoors category, nor as any statement about
recommendation difficulty or expected model performance.
