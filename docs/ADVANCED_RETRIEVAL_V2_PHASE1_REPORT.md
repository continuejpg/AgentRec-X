# Advanced Retrieval v2 — Phase 1 Report

Phase 1 was CPU/local only. No AutoDL Qwen service was used. No full BGE re-embedding was
attempted locally.

## 1. Existing Engineering Audit

Reused:

- `CatalogSearchSource` for BM25;
- existing identity-aligned BGE index at `runs/p14d2/bge_index.npz`;
- existing `LocalBGEDenseIndex` loader/encoder;
- existing CandidateLedger / Eligibility / Decision Gate;
- existing P14-E2/P14-B3 tasks and evaluator labels;
- existing MCP `run_agent_turn`;
- existing `runs/p14d2/catalogue_docs.jsonl` and BGE top-100 results.

Findings:

- `CatalogSearchSource` already had `bm25`, `bge`, and an opt-in `hybrid` mode with fixed
  RRF k=60. Phase 1 did **not** reimplement it; it added a `hybrid_rrf` alias, extracted
  a tested `reciprocal_rank_fusion()` helper, retained per-arm BM25/BGE ranks and scores,
  exposed hybrid through MCP, and tested dedup/foreign-id handling.
- `CatalogSearchSource` was not wired into the direct A/B task harness with BGE. The
  direct runner loaded the catalogue but left the default BM25 source. That is the root
  cause of the reported `p14e2_lite_0049` direct/MCP difference, not a Memory difference.
  The runner now calls `configure_catalogue(...)` explicitly.
- `0054`: with the accepted BGE source, the rule Planner selects `B08HM3MLJF` and the
  evaluator marks it successful; the prior Qwen artifact selected `B09TB1SCJP` (not
  acceptable). The correct product is recalled and BGE rank 1, so this is a
  selection/rerank outcome, not a retrieval miss. No recommendation strategy was changed
  for one task.
- `0049`: direct and MCP memory reads produce the same semantic active preferences and
  the same hard constraints. Memory IDs differ only because each run uses a fresh SQLite
  memory handle. The previous direct no-candidate result was due to BM25 default vs MCP
  BGE, not Memory replay.

## 2. Planner Empty-Action Fix

`HttpPlannerProposer.propose()` now:

- retries at most two times only for malformed action payloads
  (`planner_invalid_action`, `planner_invalid_arguments`, `planner_payload_not_object`);
- records `last_attempts` and `last_parse_errors`;
- raises an explicit error after the bounded retry;
- never falls back to the Rule Planner;
- never fabricates a successful action.

Tests cover both retry-then-success and two-invalid-actions-then-explicit-error.

## 3. BM25 + BGE Hybrid Retrieval

Existing mode retained:

```text
bm25
bge
hybrid
hybrid_rrf  (alias)
```

Hybrid behavior:

- BM25 and BGE ranked lists are fused with RRF k=60, equal weights;
- duplicate item_ids are removed;
- BM25 rank/score and BGE rank/score are retained on each hit;
- dense identities absent from the current catalogue are dropped;
- original catalogue identity is unchanged (`amazon_electronics_c4:<item_id>`);
- CandidateLedger/Gate read `score_kind=hybrid_bm25_bge`; no gate changes.

A real hybrid smoke query was run against the existing index/model; it returned
`score_kind=hybrid_bm25_bge` and both per-arm ranks.

## 4. Enriched Retrieval Documents v2

Built:

```text
data/enrichment/electronics_retrieval_docs_v2.jsonl
runs/advanced_retrieval_v2/retrieval_docs_v2_manifest.json
```

- 76,263 documents;
- 924.15 mean chars, 1,800 max;
- 569 documents with explicit metadata conflicts;
- C4 catalogue and old BGE index untouched;
- document builder extracts title/product type/attrs/features/description and a small
  allow-list of structured details;
- historical price/rating are omitted from retrieval text;
- conflicting form factors are omitted, never asserted;
- source fields and metadata version are retained.

SHA256:

```text
documents_sha256 = 98f006d1f7e727d80716e94284deddadd36558524667f98a867f37b58bd16f29
catalogue_sha256 = 14f27eedd94915ad3c5372ab3a8e13cd13f432240fd572db37e4681c49aaa1e6
sidecar_sha256   = 2b76df5b650b5d017dcb9918d871e6584b793de6c05d7dac97e9df6aef1bda07
```

New BM25 source:

- `recommendation.retrieval.EnrichedBM25Source`;
- `score_kind=bm25_enriched_v2`;
- no dense encoder loaded.

## 5. CPU Retrieval Comparison

Tasks: 135 P14-D1 natural retrieval tasks.
Target labels: one observed positive; all other products unjudged. **No exhaustive hard
constraint labels are available**, so hard-verified candidate recall is reported as
unavailable rather than fabricated. K=10/20/100; RRF k=60.

### Original query

| Method | R@10 | R@20 | R@100 | Hit@1 |
|---|---:|---:|---:|---:|
| A C4 BM25 | 5/135 = 3.7% | 10/135 = 7.4% | 20/135 = 14.8% | 1.5% |
| B C4 BGE | 19/135 = 14.1% | 27/135 = 20.0% | 66/135 = 48.9% | 3.7% |
| C Hybrid RRF | 13/135 = 9.6% | 22/135 = 16.3% | 47/135 = 34.8% | 3.0% |
| D Enriched BM25 | 17/135 = 12.6% | 23/135 = 17.0% | 45/135 = 33.3% | 3.7% |

### Rewritten query

| Method | R@10 | R@20 | R@100 | Hit@1 |
|---|---:|---:|---:|---:|
| A C4 BM25 | 9.6% | 13.3% | 22.2% | 1.5% |
| B C4 BGE | 10.4% | 14.8% | 41.5% | 2.2% |
| C Hybrid RRF | 8.9% | 12.6% | 35.6% | 1.5% |
| D Enriched BM25 | 11.9% | 16.3% | 28.9% | 3.0% |

Key negative result:

> Fixed equal-weight Hybrid RRF **does not beat BGE alone** on this one-positive task set.
> BGE is the strongest method at every K in the original-query comparison. C's rank
> movements vs B are: 13 improved / 54 worsened / 68 same (original), and
> 22 improved / 41 worsened / 72 same (rewritten).

Enriched BM25 substantially improves over the old C4 BM25 at Top-10/Top-20, but remains
below BGE. It is a viable metadata-aware lexical baseline/fallback, not a replacement for
BGE.

Concrete cases:

- `p14d1_0011` (TV base): A C4 BM25 did not retrieve the observed target in Top-10;
  D Enriched BM25 ranked it 9.
- `p14d1_0022` (external Blu-ray drive): BGE ranked target 7; Hybrid RRF ranked 30.
  Equal-weight RRF let the long BM25 tail dominate a good dense ranking.

Costs:

- A: 36.9 s total, P50 258.7 ms/query, original query;
- D: 52.0 s total, P50 370.6 ms/query;
- C fusion: ~8 ms total (BGE ranks precomputed; no query encoding);
- CPU max RSS: 2,121,344 KB ≈ 2.0 GiB;
- old BGE index 213,173,605 bytes; enriched docs 201,043,972 bytes.

## 6. Enriched BGE Index Preparation

Added:

- `experiments/build_enriched_bge_index_v2.py`;
- batch encoding via `transformers` AutoModel CLS + L2 normalization;
- resumable `.partial.npz` checkpoint;
- identity/vector alignment check;
- normalized-vector check;
- manifest with docs SHA, index SHA, model path/revision, batch parameters;
- output NPZ compatible with `LocalBGEDenseIndex`.

A 32-document CPU smoke index was built and loaded successfully:

```text
runs/advanced_retrieval_v2/bge_enriched_v2_smoke.npz
docs=32, dim=768, normalized, load OK
```

**The full 76,263-document Enriched BGE index was not built** (CPU cost too high for
Phase 1). This remains the primary Phase 2 GPU task.

## 7. Reranker Interface

Added `recommendation.retrieval.reranker`:

- `reranker_mode=existing` — no model, accepted order;
- `reranker_mode=cross_encoder` — lazy Cross-Encoder adapter, explicit failure if the
  model is unavailable;
- `RerankCandidate` records existing rank/score, reranker score and final rank;
- injectable loader/scorer for tests;
- optional `sentence-transformers` default loader documented as Phase 2 dependency;
- `experiments/advanced_retrieval/run_reranker_benchmark.py` benchmark runner.

No real Cross-Encoder inference was claimed or run. Existing-mode control on BGE Top-20
reported Hit@1 3.7%, Hit@5 8.9%, Hit@10 14.1% (the same ordering as BGE; the metric is a
control, not evidence for a reranker).

## 8. Trace Assets for Future SFT

Created:

```text
runs/advanced_retrieval_v2/sft_trace_candidates.jsonl
runs/advanced_retrieval_v2/sft_trace_manifest.json
```

- 60 rows from existing A/B traces;
- fields include query, memory snapshot, planner observations, candidates, chosen action,
  Gate feedback, outcome, label source and review status;
- `review_status=provisional_not_expert_gold`;
- no new trajectory generation;
- P18 Reserved excluded.

Caveat: the source A/B artifacts predate the direct-runner BGE fix; their search mode is
BM25. They are retained as existing trace assets, not as a current BGE/Agent quality result.

## 9. Regression

Targeted tests passed:

```text
54 passed
```

Full suite result is recorded in the final Phase 1 response and GPU handoff. P18 Frozen
Manifest is untouched.

## 10. Phase 2 Handoff

See [`ADVANCED_RETRIEVAL_V2_GPU_HANDOFF.md`](ADVANCED_RETRIEVAL_V2_GPU_HANDOFF.md).
