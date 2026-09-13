# ItemCF Baseline (Milestone 2B)

The first reference recommender for AgentRec-X: a deterministic, interpretable
cosine item-item similarity model over implicit feedback. It is a **model only** —
it produces score vectors and contains no evaluation logic. Everything is scored
through the Milestone 2A unified evaluator.

---

## 1. Module layout

| Path | Responsibility |
| --- | --- |
| `recommendation/baselines/itemcf.py` | the `ItemCF` model: fit, sparse similarities, scoring |
| `experiments/itemcf_smoke.py` | real-artifact integration smoke run (wiring only) |
| `tests/test_itemcf.py` | 44 unit tests, including the hand-computable oracle |

`recommendation/baselines/` holds models; `recommendation/evaluation/` holds the
protocol. ItemCF logic is deliberately **not** in the evaluation package, and
evaluation logic is deliberately **not** here.

---

## 2. Training-data contract

The model is fitted **once**, from the `train_history` of the evaluation-eligible
users (`sequence length >= 3`):

```
for a chronological sequence [i1, ..., i(n-2), i(n-1), in]:
    fit data = [i1, ..., i(n-2)]        train_history only
```

* The validation target `i(n-1)` and the test target `in` are **never** used to
  build item frequencies, co-occurrences, similarities or any other learned state.
* Inference uses `train_history` for validation and `train_history + [i(n-1)]` for
  test. The validation interaction may therefore be supplied as *history* at test
  time (it precedes the test target) but is never folded back into the similarity
  matrix — this milestone does not refit on train+validation.
* Users excluded by the Milestone 2A cohort (`length < 3`) never enter the fit data.

`fit_from_cohort(cases, num_items)` is the sanctioned entry point: it reads
`case.train_history` and never touches `case.validation_target` / `case.test_target`.

---

## 3. Implicit-feedback semantics

Ratings are **ignored** — presence of an interaction is the whole signal. Within a
user, an item counts **once** regardless of how many times it was consumed:

```
unique_items = unique(train_history)
```

Item ids keep the preprocessing contract: PAD `0` is not a catalog item, real items
are `1..num_items`, and ids remain opaque categorical identifiers derived from
`parent_asin`.

---

## 4. Formulas

```
freq(i)    = # training users whose unique train history contains i
cooc(i,j)  = # training users whose unique train history contains both i and j
sim(i,j)   = cooc(i,j) / sqrt(freq(i) * freq(j))      for i != j
sim(i,i)   = 0                                        (self-similarity excluded)
score(j)   = sum(sim(i, j) for i in unique(history))
```

* Similarity is **symmetric** by construction.
* No rating weighting, time decay, IUF, popularity prior or learned parameter.
* Items with no learned similarity score exactly `0.0`. There is **no popularity
  fallback** in this milestone.
* Repeated items in the *inference* history do not multiply their contribution,
  because scoring iterates the distinct history items.

### Representation

Similarities and co-occurrences are sparse: `item -> {neighbour: value}`. A dense
`num_items x num_items` matrix is never materialised. Every retained pair is stored
under **both** endpoints, which keeps `score` independent of which endpoint happened
to be iterated first; the cost is `2 x` the pair count, still far below dense.

Repeated item pairs cannot occur within one user (the history is deduplicated before
pair generation), so co-occurrence counts are exact.

---

## 5. Scorer / evaluator ownership boundary

The scorer's only output is a score vector:

```
len(scores) == num_items + 1        scores[item_id] is that item's score
```

`scores[0]` is the PAD slot. The scorer performs **no masking**: it does not remove
PAD, seen items, repeated targets, or validation/test targets. The Milestone 2A
evaluator owns PAD exclusion, seen-item masking, target retention, tie handling,
ranking and metrics, and is never duplicated or bypassed here.

Scores must be **finite**; the evaluator rejects `NaN` and `±inf` (see the
evaluation README), so ItemCF emits only finite values.

---

## 6. Determinism

Given identical training histories and configuration:

* item frequencies, co-occurrences and similarities are identical;
* score vectors are identical;
* evaluation reports are identical.

No externally observable ordering depends on set/dict iteration order: frequencies,
similarity neighbour lists and the serialised `to_dict()` payload are all emitted in
ascending item-id order, and histories are deduplicated in ascending order.
`fit_seconds` is the one intentionally variable field (wall-clock timing).

---

## 7. Usage

```python
from recommendation.baselines.itemcf import fit_from_cohort, make_scorer
from recommendation.evaluation import FullRankingEvaluator, build_cohort_from_artifacts

cases, split = build_cohort_from_artifacts(sequences_path, mappings_path)
model, stats = fit_from_cohort(cases, split.catalog_size)

evaluator = FullRankingEvaluator(num_items=split.catalog_size, k_values=(5, 10, 20))
for mode in ("validation", "test"):
    outcome = evaluator.evaluate(cases, make_scorer(model), mode=mode)
    print(outcome.report.format())
```

### Real-artifact smoke run

```bash
.venv/bin/python -m experiments.itemcf_smoke
.venv/bin/python -m experiments.itemcf_smoke --json /tmp/itemcf_smoke.json
```

It fits on eligible users' train histories, evaluates validation and test through the
existing evaluator, and asserts the metric sanity checks (finite, in `[0,1]`,
`HR@K == Recall@K`, monotone in `K`, case counts equal to the cohort size).

---

## 8. Interpretation rule

The preprocessing fixture is a 100k-record *prefix* of Sports and Outdoors, not the
full category. Every metric produced from it is an **engineering smoke result**: it
demonstrates that the baseline, the split and the evaluator fit together correctly.
It is **not** a benchmark result and must not be quoted as ItemCF quality on
Sports and Outdoors, nor compared against future models as if it were a baseline
number. Real baseline numbers require the full category and the documented protocol.
