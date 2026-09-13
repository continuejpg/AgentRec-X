# Training (Milestone 4)

Minimum infrastructure to prove the Milestone 3 SASRec can actually **learn**:
a training objective, deterministic batching, and a small trainer. No scheduler,
no mixed precision, no distributed training, no checkpoint management, no early
stopping, no experiment tracking, no hyperparameter search.

| Path | Responsibility |
| --- | --- |
| `losses.py` | the binary logistic objective and logit diagnostics |
| `sasrec.py` | `TrainerConfig`, batching, `SASRecTrainer`, `TrainingResult` |
| `tests/test_sasrec_training.py` | 45 unit tests incl. the tiny overfit fixture |
| `tests/test_sasrec_training_integration.py` | 12 real-artifact / evaluator-reuse tests |
| `experiments/sasrec_tiny_overfit.py` | tiny overfit + bounded real smoke runner |

---

## 1. Responsibility split

```
Dataset    -> training examples / negatives
Model      -> hidden states / logits / scores
Trainer    -> loss / backward / optimizer        (this package)
Evaluator  -> masking / ranking / HR / Recall / NDCG
```

The trainer never masks candidates, never ranks and never computes a metric; it has
no `rank`/`mask`/`ndcg`/`hr_at_k` API (asserted by test). Evaluation semantics are
not duplicated anywhere.

## 2. Loss

For aligned valid-position logits (`p` positive, `n` negative):

```
per_position = softplus(-p) + softplus(n)
loss         = mean(per_position)
```

`softplus` is `log1p(exp(-|x|)) + max(x, 0)` — stable, and not thresholded to the
`x` approximation. Zero logits give exactly `2·log(2)` (float32 precision).

* Padding contributes **nothing**: `training_logits` already drops padding
  positions, so no fake PAD labels exist anywhere.
* Positive and negative logits must have identical, non-empty shape.
* An empty position set raises `TrainingLossError` rather than returning NaN.
* **Non-finite logits fail fast.** Before any loss arithmetic,
  `ensure_finite_logits` requires `torch.isfinite(...).all()` on both tensors and
  raises `TrainingLossError` (a `ValueError`) naming the offending side, the count
  and the first bad index. Nothing is clamped, replaced, `nan_to_num`-ed, ignored or
  averaged around.

The fail-fast guard exists for a specific reason: stable `softplus` **saturates**,
so an infinite *positive* logit yields `softplus(-inf) + softplus(0) = log 2` — a
plausible-looking finite loss. That mathematical saturation must not be allowed to
hide a numerical model failure, so `NaN`, `+Inf` and `-Inf` are rejected outright.
For **finite** inputs the arithmetic is unchanged (`2·log 2` for zero logits,
`0.0` for `p = 1e4, n = -1e4`).

## 3. Batching

Hand-rolled (`num_workers = 0` semantics by construction) rather than
`torch.utils.data.DataLoader`: samples are already fixed-length and left-padded, so
collation is a pure stack, and this keeps ordering explicit. No multiprocessing.

* batch tensors are `torch.long`, shape `[batch, max_seq_len]`;
* input/positive/negative stay aligned and left-padded (verified against the source);
* `drop_last` is deliberately **not** used — the final partial batch is emitted,
  because dropping it would discard real transitions from tiny fixtures;
* sample order comes from `epoch_order(n, shuffle, seed, epoch)`, seeded from
  `(seed, epoch)` only, so it is independent of global RNG state and of
  `PYTHONHASHSEED`;
* zero-transition users produce no samples, so an all-PAD gradient sample is
  impossible.

## 4. Optimum / configuration

`TrainerConfig`: `learning_rate`, `weight_decay` (default `0.0` — no regularisation
needed to memorise a tiny fixture, exposed for later milestones), `batch_size`,
`epochs`, `seed`, `device` (**must be `"cpu"`** in this milestone), `shuffle`,
`resample_negatives`, optional `max_grad_norm`.

Optimizer: **AdamW**, built over parameters with `requires_grad`.

Per step: `zero_grad(set_to_none=True)` → forward → loss → `backward()` →
(optional clip) → `step()`. After every step the item embedding's PAD row is
re-asserted to exactly zero, because `padding_idx` only zeroes the gradient *into*
PAD while AdamW's weight-decay term can still move the row.

## 5. Epoch-aware negative sampling

`resample_negatives=True` redraws negatives each epoch through the Milestone 3
sampler, keyed on `(seed, epoch, user, position)`. Inputs and positives are copied
through untouched, so the structural optimisation target is unchanged while the
negatives are refreshed. Validation/test targets remain invisible. Determinism is
covered by tests, and an exhausted pool resolves through the deterministic fallback
(never spins).

## 6. Usage

```python
from recommendation.training import TrainerConfig, train_sasrec

config = TrainerConfig(learning_rate=0.01, batch_size=32, epochs=2, seed=0,
                       resample_negatives=True)
result = train_sasrec(model, dataset, config)
print(result.initial_loss, result.final_loss, result.loss_reduction_ratio)
```

### Runners

```bash
.venv/bin/python -m experiments.sasrec_tiny_overfit          # exit 0 if criteria pass
.venv/bin/python -m experiments.sasrec_tiny_overfit --json out.json
```

## 7. Interpretation rule

The tiny fixture demonstrates **memorisation**, not recommendation quality. The
bounded real-artifact smoke is a pipeline integration check on a 100k prefix
fixture: it is **not** Milestone 5 training, **not** a benchmark, and any metric the
frozen evaluator computes during it is a smoke diagnostic only. `max_seq_len = 20`
is the existing development setting, not a justified benchmark value.
