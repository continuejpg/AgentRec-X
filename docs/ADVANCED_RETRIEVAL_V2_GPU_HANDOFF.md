# Advanced Retrieval v2 — Phase 2 GPU Handoff

## A. Git State

- Phase 1 started at: `3adc04c` (`docs: add reproducible evaluation requirements`)
- Phase 1 code/test changes: see the `HEAD` printed by `git log -1` in the final Phase 1
  session response. This handoff file and the Phase 1 report are part of that Phase 1
  commit sequence.
- Worktree expected: clean after the Phase 1 commit.
- `runs/` and model/data artifacts are deliberately not committed.
- P18 Frozen Manifest was not touched.

## B. Data & Index Manifest

| Asset | Path | SHA256 / version |
|---|---|---|
| C4 catalogue | `data/p13r/electronics/catalog_meta.jsonl` | `14f27eedd94915ad3c5372ab3a8e13cd13f432240fd572db37e4681c49aaa1e6` |
| Metadata sidecar | `data/enrichment/electronics_metadata_sidecar.jsonl` | `2b76df5b650b5d017dcb9918d871e6584b793de6c05d7dac97e9df6aef1bda07` |
| Enriched retrieval docs v2 | `data/enrichment/electronics_retrieval_docs_v2.jsonl` | `98f006d1f7e727d80716e94284deddadd36558524667f98a867f37b58bd16f29` |
| Docs manifest | `runs/advanced_retrieval_v2/retrieval_docs_v2_manifest.json` | 76,263 documents, 569 conflict docs |
| Original BGE index | `runs/p14d2/bge_index.npz` | `fce84987083452abd84b7fdbf09b1d55e35305e7c4983b03343034dca3f7a69d` |
| BGE model | `models/bge-base-en-v1.5` | revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| Enriched BGE smoke index | `runs/advanced_retrieval_v2/bge_enriched_v2_smoke.npz` | 32 docs, SHA `ef167b5aa8685db61f7294305ec0e06a74e5c8a7895ea91de360e9c6de3ac709` |
| Full enriched BGE index | **NOT BUILT** | expected `runs/advanced_retrieval_v2/bge_enriched_v2.npz` |
| Old C4 BGE retrieval ranks | `runs/p14d2/bge_retrieval.json` | real existing top-100 asset |

The enriched docs are additive. The C4 catalogue and original BGE index were not
overwritten.

## C. GPU Tasks Remaining

1. **Enriched BGE embedding** — encode 76,263 enriched documents with the pinned BGE
   model and produce the full NPZ + manifest.
2. **Cross-Encoder reranker** — real BGE-Reranker or Qwen3-Reranker inference. No real
   reranker was run in Phase 1.
3. **Retrieval A/B with enriched BGE and Hybrid RRF** — compare A C4 BM25, B C4 BGE,
   C C4 Hybrid RRF, D Enriched BM25, and the new Enriched BGE / Hybrid on the fixed
   135 P14-D1 queries.
4. **Real Qwen Agent A/B** — Baseline vs Corrective vs Complete Shadow with the selected
   retrieval mode. Phase 1 did not run remote Qwen Live. The prior direct A/B artifacts
   used BM25 by mistake; the Phase 1 runner now calls `configure_catalogue` and accepts
   `--search-mode`, but the Qwen arm must be rerun in Phase 2.

## D. Commands

All commands run from the repository root with the accepted `.venv`.

Install the client/eval and optional model runtime dependencies first:

```bash
.venv/bin/python -m pip install -r requirements-eval.txt
# The advanced-retrieval file is included by requirements-eval.txt.
```

### 1. Rebuild enriched docs (CPU, already built)

```bash
PYTHONPATH=. .venv/bin/python experiments/build_enriched_retrieval_docs.py
```

### 2. Build the full enriched BGE index on GPU

```bash
PYTHONPATH=. .venv/bin/python experiments/build_enriched_bge_index_v2.py \
  --docs data/enrichment/electronics_retrieval_docs_v2.jsonl \
  --out runs/advanced_retrieval_v2/bge_enriched_v2.npz \
  --manifest runs/advanced_retrieval_v2/bge_enriched_v2_manifest.json \
  --model /root/autodl-tmp/models/bge-base-en-v1.5 \
  --model-revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
  --device cuda --batch-size 64 --checkpoint-every 20
```

If interrupted, rerun with `--resume`; the partial NPZ prefixes the same sorted document
order and is validated before continuing.

### 3. Retrieval comparison with the actual dense index

```bash
PYTHONPATH=. .venv/bin/python experiments/advanced_retrieval/run_cpu_retrieval_comparison.py \
  --bge-index runs/advanced_retrieval_v2/bge_enriched_v2.npz \
  --bge-model /root/autodl-tmp/models/bge-base-en-v1.5 \
  --variants original rewritten
```

Without `--bge-index` it uses the real precomputed C4 BGE top-100 ranks for the Phase 1
control. `--bge-index` performs actual dense query encoding through
`LocalBGEDenseIndex`; on a CPU-only host this is slower.

### 4. Reranker benchmark (real model is Phase 2)

```bash
PYTHONPATH=. .venv/bin/python experiments/advanced_retrieval/run_reranker_benchmark.py \
  --mode cross_encoder \
  --model /root/autodl-tmp/models/bge-reranker-v2-m3 \
  --device cuda --batch-size 8 --depth 20
```

The default CrossEncoder loader uses `sentence-transformers`; install the optional
package or inject a model-specific loader/scorer. A missing model raises an explicit error.

Existing-order control (already run):

```bash
PYTHONPATH=. .venv/bin/python experiments/advanced_retrieval/run_reranker_benchmark.py \
  --mode existing --depth 20
```

### 5. MCP with an alternative dense index

```bash
export AGENTRECX_QWEN_PLANNER=1
export AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029
export AGENTRECX_QWEN_PLANNER_MODEL=Qwen3.5-9B-Base
export AGENTRECX_QWEN_EXPLAIN=1

.venv/bin/python -m recommendation.mcp.server \
  --user-key p14e2_lite_0049 \
  --memory-fixture data/p14e2/benchmark/dev.agent.jsonl \
  --search-mode hybrid \
  --dense-index runs/advanced_retrieval_v2/bge_enriched_v2.npz \
  --dense-model /root/autodl-tmp/models/bge-base-en-v1.5 \
  --metadata-mode enriched \
  --corrective-rag --evidence-verifier
```

`--search-mode` accepts `bm25`, `bge`, `hybrid`, and `hybrid_rrf` (alias for `hybrid`).
The MCP config records the selected mode; dense failures fall back explicitly to BM25
with a warning, never to fabricated dense hits.

### 6. Real Qwen Agent A/B after GPU retrieval is ready

```bash
export AGENTRECX_QWEN_PLANNER=1
export AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029
export AGENTRECX_QWEN_PLANNER_MODEL=Qwen3.5-9B-Base
export AGENTRECX_QWEN_PLANNER_TIMEOUT_S=300

PYTHONPATH=. .venv/bin/python experiments/p20_agent_eval/run_task_arms.py \
  --policies baseline corrective \
  --search-mode bge \
  --dense-index runs/advanced_retrieval_v2/bge_enriched_v2.npz \
  --dense-model /root/autodl-tmp/models/bge-base-en-v1.5
```

Then the representative Complete Shadow subset:

```bash
PYTHONPATH=. .venv/bin/python experiments/p20_agent_eval/run_complete_shadow.py \
  --dense-index runs/advanced_retrieval_v2/bge_enriched_v2.npz \
  --dense-model /root/autodl-tmp/models/bge-base-en-v1.5
```

### 7. Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_advanced_retrieval.py tests/test_reranker_interface.py \
  tests/test_planner_contract.py tests/test_bge_performance.py \
  tests/test_p14a_catalogue_search.py tests/test_p17_mcp.py \
  tests/test_planner_reliability.py tests/test_evidence_verifier.py
```

Targeted Phase 1 result: **54 passed**.

The full suite attempt exceeded the Phase 1 time budget and was stopped at ~90%; it
reported several failures that were not triaged as Phase 1 regressions. Do not claim a
clean full-suite result in Phase 2 without rerunning it.

## E. Reproduction Rules

- Retrieval task set: **135 P14-D1 natural queries**; original and rewritten query
  variants are separate.
- Relevance: one observed positive per task. Other catalogue items are **unjudged**, not
  negatives. No human relevance labels.
- Retrieval metrics: Recall@{10,20,100}, Hit@1, target coverage; hard-verified candidate
  recall is unavailable and must stay null unless a hard-constraint label set is created.
- Hybrid: fixed equal-weight RRF, **k=60**; no tuning on the evaluation targets.
- Enriched BM25: `EnrichedBM25Source`, `score_kind=bm25_enriched_v2`.
- Agent task set: 30 P14-E2/P14-B3 natural tasks; 10-task Complete Shadow subset.
- Memory: task-scoped fixture (`dev.agent.jsonl` or `eval100.agent.jsonl`), fresh
  `ShoppingCompanionMemory` per task; no caller-supplied user id in MCP.
- Gate: existing Decision Gate remains authoritative; retrieval mode cannot change Gate
  logic, hard constraints, UNKNOWN handling or product identity.
- Qwen generation: existing model service, `/plan` and `/explain`, `max-new-tokens 512`;
  record token counts and latency from the response.
- Verifier: shadow only; no automatic interception; P18 Reserved excluded.
- GPU: one Qwen/BGE model instance at a time; the previous observed combined peak was
  ~23.7 GiB, leaving very little headroom. Do not run the 9B Qwen service and a 9B
  reranker concurrently without an explicit VRAM plan.
