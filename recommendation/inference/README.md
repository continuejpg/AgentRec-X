# Inference and serving-time ranking (Milestone 6)

Framework-independent components that turn the accepted SASRec checkpoint into
recommendations. The HTTP service in [`../api/`](../api/) is a thin adapter over this
package, and future callers (for example an Agent recommendation tool) should use
these classes directly rather than going through HTTP.

| Module | Responsibility |
| --- | --- |
| `sasrec.py` | checkpoint loading + identity validation, item mapping, history encoding, forward pass, `recommend()` |
| `ranking.py` | PAD/seen masking and the deterministic top-k rule |

## Score meaning

`score_catalog()` returns **raw SASRec model scores** (length `num_items + 1`). They are
not probabilities and are only meaningful for ordering.

## Production ranking differs from evaluation ranking

| | Evaluator (Milestone 2A) | Serving (here) |
| --- | --- | --- |
| Positive target | one held-out target | none |
| Target retention | target stays eligible even if seen | n/a |
| Candidate set | catalog − seen, target retained | catalog − seen |
| Tie rule | higher score, then lower item id | same |

The evaluator remains the single source of truth for *benchmark* metrics; this layer
never computes HR/Recall/NDCG.

## Encoding vs. masking

Two different history concepts, deliberately kept apart:

* **model window** — the newest `max_seq_len` items, left-padded with PAD 0; what the
  Transformer sees;
* **full supplied history** — every item the caller sent; drives seen-item masking so a
  previously interacted item is never recommended.

## Inference mode

The engine freezes all parameters (`requires_grad=False`), keeps the model in
`eval()`, and runs every forward under `torch.inference_mode()`. No optimizer exists.

## Quick start

```python
from recommendation.inference import InferenceConfig, SASRecInferenceEngine

engine = SASRecInferenceEngine(InferenceConfig(
    checkpoint_path="runs/sasrec_canonical_2026/best.pt",
    mappings_path="data/processed/Sports_and_Outdoors_mappings.json",
    manifest_path="runs/sasrec_canonical_2026/run.json",
    device="cpu",
))
result = engine.recommend(["B00...", "B01..."], k=10)
for item in result.recommendations:
    print(item.rank, item.parent_asin, item.score)
```
