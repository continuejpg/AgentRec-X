# Inference and serving-time ranking (Milestone 6)

Framework-independent components that turn the accepted SASRec checkpoint into
recommendations. The HTTP service in [`../api/`](../api/) is a thin adapter over this
package, and future callers (for example an Agent recommendation tool) should use
these classes directly rather than going through HTTP.

| Module | Responsibility |
| --- | --- |
| `sasrec.py` | checkpoint loading + identity validation, item mapping, history encoding, forward pass, `recommend()` |
| `ranking.py` | PAD/seen masking and the deterministic top-k rule |
| `twotower.py` | the Two-Tower / DSSM retrieval engine (`score_kind = two_tower_cosine`) |
| `lexical.py` | whole-catalogue BM25 source (`score_kind = bm25`), sharing the tokenizer/IDF/BM25 constants with [`../rag/retrieval.py`](../rag/retrieval.py) |
| `fusion.py` | **Serving V1**: a fixed multi-source `RecommendationEngine` fusing SASRec, Two-Tower and the lexical source with the accepted `reciprocal_rank_fusion` at `RRF_K = 60` |

## Serving V1 (fixed multi-source serving)

`FusionInferenceEngine` implements the same `RecommendationEngine` contract as
`SASRecInferenceEngine` (`recommend(history_parent_asins, k) -> RecommendationResult`), so the Tool,
the agent graph, the catalogue enricher, the preference matcher and the preference reranker are all
unchanged: the engine decides *which* candidates exist and in what fused order, and nothing else.

* **Fixed, not adaptive.** The same three sources run for every request, in the same order, with the
  repository's constant — no source selection, no weighting, no raw-score fusion. The sources score
  on incomparable scales, so only their orderings are combined.
* **Scores are labelled.** A fused score is not a model score: `score_kind` is `rrf_rank_fusion`, and
  per-item provenance records each source's own rank and score kind.
* **Bounded by the accepted contract.** Each source is read to `DEFAULT_SOURCE_HEAD = MAX_K = 100`,
  the deepest ranking the serving engines permit. The offline fusion experiments read depth-1000 head
  lists out of a score matrix; that difference is real and is stated rather than papered over.
* **Failure is explicit.** An empty history, an identity outside the catalogue, or a fused identity
  the trusted side cannot confirm fails or is dropped — never silently substituted.

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
