# SASRec Dataset & Model (Milestone 3)

Milestone 3 adds the SASRec **data contract** and **architecture**. There is no
training here: no optimizer, no epoch loop, no loss step, and no benchmark metric.

| Path | Responsibility |
| --- | --- |
| `recommendation/datasets/sasrec.py` | training sequences, truncation, negative sampling, inference encoding |
| `recommendation/models/sasrec.py` | the SASRec architecture, training logits, full-catalog scoring |
| `tests/test_sasrec_dataset.py` | 41 dataset tests |
| `tests/test_sasrec_model.py` | 43 model tests (incl. causal isolation) |
| `tests/test_sasrec_integration.py` | real-artifact data-contract checks |
| `experiments/sasrec_smoke.py` | real-artifact dataset + CPU forward smoke run |

Datasets live in `datasets/`, models in `models/`, and the evaluation protocol stays
in `evaluation/` — SASRec contains no evaluation logic.

---

## 1. Training-data contract

The only permitted input is the Milestone 2A evaluation cohort's
`EvaluationCase.train_history`:

```
for [i1, ..., i(n-2), i(n-1), in]  with n >= 3:
    train_history     = [i1, ..., i(n-2)]   <-- the ONLY training input
    validation_target = i(n-1)              <-- held out
    test_target       = in                  <-- held out
```

`validation_target` and `test_target` are never consulted — not for sequence
construction, not for truncation decisions, not for negative-sampling exclusion, not
for statistics. `build_dataset` takes `EvaluationCase` objects and reads only
`train_history`, so the guarantee is structural: there is no code path that reads a
target. `cohort_structure_digest()` hashes train histories only, and the tests prove
that changing every target leaves the dataset digest byte-identical.

Excluding a *future* target from negatives would itself be leakage: at training time
that item is simply an unseen item.

### Trainable-user policy

Evaluation eligibility is `sequence length >= 3`, so every evaluation user has
`len(train_history) >= 1`. A SASRec sample needs at least one transition, i.e.
`len(train_history) >= 2`:

* `len(train_history) == 1` → **zero transitions**, contributes no gradient sample;
* that user is **not** removed from the validation/test evaluation cohort — the
  exclusion is dataset-level only, and is reported as `users_with_zero_transitions`
  and `users_with_one_train_item`.

---

## 2. Sequence construction

For a history `[h1, ..., hm]` with `m >= 2`:

```
input_ids    = [h1, ..., h(m-1)]
positive_ids = [h2, ..., hm]
```

* Arrays are exactly `max_seq_len` long.
* **Left padding** with PAD `0`. A history of `m` items yields `m-1` real slots.
* `positive_ids == 0` marks a padding position — later training excludes those
  positions via `valid_position_mask`.
* Truncation keeps the **most recent** transitions and is applied to the shifted
  arrays together, so an input and its next item never drift apart:

```
history = [1,2,3,4]  max_seq_len = 5      history = [1,2,3,4]  max_seq_len = 3
  input_ids    = [0,0,1,2,3]                input_ids    = [1,2,3]
  positive_ids = [0,0,2,3,4]                positive_ids = [2,3,4]
```

Source sequences are never mutated (arrays are tuples).

### Inference encoding

Training construction and inference encoding are deliberately different functions.
For validation `history = train_history`; for test
`history = train_history + [validation_target]` (the validation interaction precedes
the test target, so it is legitimate history). Encoding keeps the most recent
`max_seq_len` items, left-pads shorter histories, preserves chronological order,
rejects an empty history and invalid ids, and never appends or inspects a target.

---

## 3. Negative sampling

One negative per non-padding positive position:

* in `1..num_items`, never PAD;
* never in the user's **train history** exclusion set;
* never influenced by validation/test targets.

Sampling is a pure function of
`(user_int_id, position, epoch, seed, exclusion)`: the RNG seed is
`sha256("agentrecx.sasrec.neg|seed|epoch|uid|position")`, so results are reproducible
and can be resampled per epoch by bumping `config.epoch` (no training loop is
implemented). The candidate id is drawn by rejection sampling from the full id range;
if `negative_retry_limit` draws all land inside the exclusion set, a deterministic
fallback picks a uniformly random offset into the complement. An empty pool raises
`SASRecDataError` — it never spins.

---

## 4. Architecture

Block order is **pre-normalization**, documented explicitly:

```
x = item_embedding(input_ids) + positional_embedding[0..L-1]
x = dropout(x)
for each block:
    x = x + residual_dropout(causal_self_attention(attention_norm(x)))
    x = x + residual_dropout(feed_forward(feed_forward_norm(x)))
x = final_norm(x)
```

* item embedding of `num_items + 1` rows with `padding_idx=0`;
* learned positional embeddings up to `max_seq_len`;
* multi-head scaled dot-product attention, written out explicitly:
  `scores = qkᵀ/√d → masked_fill(blocked, -inf) → softmax → dropout → ·v`;
* point-wise feed-forward `Linear → GELU → dropout → Linear`;
* residual connections, layer normalization, dropout, configurable `num_blocks`,
  final layer normalization.

Deliberately excluded: side information, ratings, text/category/RAG features, user
embeddings, time decay, or any positional scheme beyond learned positional
embeddings.

### Masks

`build_causal_attention_mask` returns a `[batch, 1, seq, seq]` **True = blocked**
mask combining two rules:

* **causality** — query `t` cannot attend to key `t' > t`;
* **padding** — padded positions are blocked as keys, and padded *query* rows are
  blocked entirely, so their hidden states are exactly zero rather than a copy of a
  real item's representation.

`padding_idx=0` keeps PAD's embedding row at zero, and `encode` re-multiplies by the
valid mask so padded rows are zero by construction.

> Note: `torch.nn.functional.scaled_dot_product_attention` returned `NaN` rows for
> this mask layout on the pinned CPU build (verified against a manual reference), so
> the attention steps are written out explicitly. The manual form is also directly
> inspectable by the tests.

### Interfaces

```python
positive_logits, negative_logits = model.training_logits(input_ids, positive_ids, negative_ids)
scores = model.full_catalog_scores(input_ids)     # [batch, num_items + 1]
```

* `training_logits` returns **flat** tensors over valid positions only, aligned with
  each other; padding positions are dropped, never silently given a loss.
  `positive_logit[t] = dot(hidden[t], item_embedding[positive_ids[t]])`, same for
  negatives, using the shared item embedding table.
* `full_catalog_scores` uses the hidden state at the **last valid position** and the
  same (weight-tied) embedding table. `scores[b, item_id]` is that item's score.

### Scorer/evaluator boundary

`make_score_fn` adapts the model to the Milestone 2A scorer signature. The model does
**no** evaluation masking: it does not remove seen items, does not remove PAD, does
not treat targets specially, does not apply tie handling, and does not compute
HR/Recall/NDCG. The PAD column is finite (its embedding row is zero), so the
evaluator's whole-vector finiteness check passes.

---

## 5. Usage

```python
from recommendation.datasets.sasrec import SASRecDatasetConfig, build_dataset
from recommendation.evaluation import build_cohort_from_artifacts
from recommendation.models import build_model, make_score_fn
from recommendation.evaluation import FullRankingEvaluator

cases, split = build_cohort_from_artifacts(sequences_path, mappings_path)
dataset = build_dataset(cases, split.catalog_size, SASRecDatasetConfig(max_seq_len=50, seed=0))

model = build_model(num_items=split.catalog_size, seed=0, max_seq_len=50)

# later (Milestone 4): training_logits + an optimizer.  Not implemented here.
evaluator = FullRankingEvaluator(num_items=split.catalog_size, k_values=(5, 10, 20))
outcome = evaluator.evaluate(cases, make_score_fn(model, 50, split.catalog_size), mode="test")
```

### Smoke run

```bash
.venv/bin/python -m experiments.sasrec_smoke
.venv/bin/python -m experiments.sasrec_smoke --json /tmp/sasrec_smoke.json
```

No training, no optimizer, no epoch loop, and no quality metric.

---

## 6. Interpretation rule

The preprocessing fixture is a 100k-record *prefix* of Sports and Outdoors.
`max_seq_len = 20` in the smoke runner is a **development integration setting**
chosen for speed — it is not a justified production/benchmark value and must not be
derived from this prefix distribution. No metric produced from an untrained model is
meaningful, and none is reported.
