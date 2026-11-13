# Advanced Retrieval v2 — Phase 2 GPU Experiments & Qwen Agent Evaluation

GPU pass completed on the AutoDL RTX 4090 D instance (`ssh -p 36856`), using the existing
Qwen3.5-9B-ms and BGE-base-en-v1.5 models. No model re-download was needed.

## 1. GPU Environment & Handoff Verification

- Remote project path: `/root/autodl-tmp/AgentRec-X-phase2`
- Remote code was synced from the local Phase 1 tree; the remote directory is **not a Git
  checkout**. Local Git history/commits remain authoritative.
- CUDA: `torch 2.11.0+cu128`, GPU RTX 4090 D, 24,564 MiB total.
- Existing model assets:
  - `/root/autodl-tmp/models/Qwen3.5-9B-ms`
  - `/root/autodl-tmp/models/bge-base-en-v1.5`
  - `/root/autodl-tmp/models/Qwen3-Reranker-4B`
- Original catalogue hash and original BGE index hash matched the local/Phase 1 values.
- Metadata sidecar was transferred; enriched documents were rebuilt on the GPU host.
- No upstream Parquet dataset or Qwen/BGE base model was downloaded.
- Remote disk after all passes: **3.4 GB free** (94% used).
- Qwen service and GPU sampler were stopped after the final pass; GPU idle at 1 MiB.

## 2. Pass A — Enriched BGE Full Index

Built with `experiments/build_enriched_bge_index_v2.py` on CUDA, batch 64, max length 512:

| Item | Value |
|---|---:|
| Documents | 76,263 |
| Index size | 215 MB |
| Build time | 857.7 s (14.3 min) |
| Embedding dim | 768 |
| Model revision | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| Index SHA256 | `0c394f28bb3d87f4baf4cc5ac816d2620275424168c3f347a6cb803e8aee2bd3` |
| Docs SHA256 | `98f006d1f7e727d80716e94284deddadd36558524667f98a867f37b58bd16f29` |
| Observed index GPU memory | ~2,626 MiB |
| Alignment | identities/vectors equal; all vectors L2-normalized |

Token check:

```text
mean 226.39 tokens; p50 202; p95 445; max 1154
documents truncated at 512: 1,384 / 76,263 = 1.8148%
```

The document builder places Title, Product Type, structured Attributes and Specifications
before long features/description, so truncation primarily affects the tail of long
marketing descriptions, not the key structured fields. This is recorded, not hidden.

## 3. Retrieval Before/After

Fixed 135 P14-D1 tasks; one observed positive per task; other items unjudged.
A=Original BGE, B=Enriched BGE, C=Enriched BGE + Enriched BM25 RRF, D=Original BGE + Original BM25 RRF.

### Original query

| Method | Hit@1 | Hit@10 | Hit@20 | Hit@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| A Original BGE | 3.7% | 14.1% | 20.0% | 48.9% | 0.0628 | 0.0809 |
| B Enriched BGE | **5.9%** | **24.4%** | **34.1%** | **55.6%** | **0.1069** | **0.1391** |
| C Enriched + BM25 RRF | 8.1% | 20.7% | 28.9% | 52.6% | 0.1226 | 0.1428 |
| D Original + BM25 RRF | 3.0% | 9.6% | 16.3% | 34.8% | 0.0498 | 0.0609 |

### Rewritten query

| Method | Hit@1 | Hit@10 | Hit@20 | Hit@100 |
|---|---:|---:|---:|---:|
| A Original BGE | 2.2% | 10.4% | 14.8% | 41.5% |
| B Enriched BGE | 3.7% | 17.0% | 23.0% | 52.6% |
| C Enriched RRF | 3.7% | 14.8% | 23.0% | 46.7% |
| D Original RRF | 1.5% | 8.9% | 12.6% | 35.6% |

Conclusion:

- **Enriched BGE is a real retrieval improvement** over original BGE on this task set.
- Enriched BGE loses the original target in **8/135** original-query cases and **3/135**
  rewritten cases, but gains many more; net Top-10/20/100 improves strongly.
- Equal-weight Hybrid RRF still does **not** beat Enriched BGE at Top-10/20/100. It improves
  Top-1 only, at the cost of candidate coverage. It is not selected.
- D confirms the old Hybrid formulation remains worse than dense-only.

Examples:

- `p14d1_0009`: target duplicate-charging cable, A rank 97 → B rank 2.
- `p14d1_0011`: TV base target, A not in Top-100 → B rank 5.
- `p14d1_0024`: AirPods Pro target, A not in Top-100 → B rank 7.
- `p14d1_0034`: Bose NC700 target, A rank 77 → B outside Top-100 (case of lost target).

Costs:

- B enriched query encoding: 66.2 s original / 53.7 s rewritten (CPU query encoding);
- C fusion plus enriched BM25: ~161 s total original;
- CPU max RSS: 3,238,196 KB ≈ 3.1 GiB;
- old BGE index 204 MB, new index 215 MB, docs 192 MB.

## 4. Pass B — Real Qwen3-Reranker-4B

Reused existing **Qwen3-Reranker-4B** (Apache-2.0, locally present, 4-bit NF4), rather than
downloading another model.

Fixed candidate pool: original BGE Top-20 per task, 135 tasks = 2,700 pairs.

| Metric | Before | After reranker |
|---|---:|---:|
| Target in candidate pool | 27/135 | 27/135 |
| Hit@1 | 5/135 = 3.7% | **15/135 = 11.1%** |
| Hit@5 | 12/135 = 8.9% | **23/135 = 17.0%** |
| Hit@10 | 19/135 = 14.1% | **27/135 = 20.0%** |
| Mean target rank | 7.22 | **2.48** |
| Improved / worsened / unchanged | — | **20 / 3 / 112** |

Latency/resources:

- total rerank benchmark: 99.2 s; model load 20.1 s; scoring 79.1 s;
- per-task P50 562 ms, P95 741 ms (20 candidates/task);
- 4-bit Qwen3-Reranker allocated peak: 4,037 MiB;
- observed nvidia-smi peak during reranker stage: ~9,774 MiB;
- no concurrent Qwen9B and reranker.

Important distinction:

> Target-in-pool is only 27/135. Reranker improves **ranking of already-retrieved**
> candidates; it cannot improve retrieval recall. It is a **reranking gain**, not a
> retrieval recall gain. It was not integrated into the Qwen Agent in this phase.

## 5. Pass C — Real Qwen Agent A/B

Qwen3.5-9B Base, same checkpoint, same 30 natural tasks, baseline policy, corrective RAG
off, verifier shadow off, BGE source explicitly configured.

| Metric | A Original BGE | B Enriched BGE |
|---|---:|---:|
| Task success | **29/30 = 96.7%** | 27/30 = 90.0% |
| Recommend success / Top-1 | **18/19** | 16/19 |
| ASK / no-match / abstain | 11/11 | 11/11 |
| Memory-conflict ASK | 2/2 | 2/2 |
| Hard-constraint violations | 0 | 0 |
| Recommend candidate recall (any verified in retrieved pool) | **19/19** | 18/19 |
| Planner calls | 102 | 101 |
| Actual search calls | 32 | 33 |
| Input / output tokens | 157,070 / 8,941 | 155,561 / 9,325 |
| Agent latency | 508.9 s | 534.8 s |
| Reranker calls | 0 | 0 |

### Failures and attribution

- A: `0051` failed → selected `B07BKPKXTT`; an acceptable candidate was in the retrieved
  pool, so this is a **selection** failure.
- B: `0045` failed → selected `B08JH9XHWC`; acceptable candidates were in the pool.
  **selection** failure after retrieval change.
- B: `0065` failed → selected `B09C1GQ3KL`; acceptable candidates were in the pool.
  **selection** failure.
- B: `0051` failed → retrieval pool had **no verified candidate**; this is the one
  candidate-recall loss from the new retrieval.

Specific questions:

- `0049`: A selected `B097PNMZ42` (success); B selected `B0CDP93K8S` (success). No
  benefit, no regression.
- `0054`: A selected `B08HM3MLJF` (success); B selected `B07ZF8893F` (success). The
  previous top-1 problem is no longer reproduced with the fixed BGE configuration; the
  retrieval change did not create a measurable gain for this task.
- Multi-hard-constraint/no-match: ASK/no-match correctness stayed 11/11 in both arms; no
  hard violations.

## 6. Adoption Decision

**Adopt for the Qwen Planner SFT baseline: Original BGE.**

- Enriched BGE has clearly better retrieval metrics, but the **Agent end-to-end task
  success decreased** from 29/30 to 27/30. The regression is concentrated in Qwen
  selection/decision, with one candidate-recall loss.
- Equal-weight Hybrid RRF is **rejected** as a default: it loses Top-10/20/100 versus
  Enriched BGE.
- Qwen3-Reranker-4B is **experimental only**: strong reranking gain on a fixed old-BGE
  pool, but not integrated into Agent, adds ~0.56 s/task and 4 GB allocated VRAM, and
  cannot fix missing candidates.
- Original BGE remains the safe baseline with the strongest observed Qwen Agent success
  and zero hard-constraint violations.

This is a deliberate rejection of “newer retrieval is automatically better.”

## 7. SFT Retrieval Baseline (Fixed)

For the next QLoRA Planner SFT round:

- Retrieval backend: **Original BGE**
- Original BGE index: `runs/p14d2/bge_index.npz`
  (`fce84987083452abd84b7fdbf09b1d55e35305e7c4983b03343034dca3f7a69d`)
- BGE model revision: `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- Metadata: enriched sidecar
  (`2b76df5b650b5d017dcb9918d871e6584b793de6c05d7dac97e9df6aef1bda07`)
- Catalogue: `14f27eedd94915ad3c5372ab3a8e13cd13f432240fd572db37e4681c49aaa1e6`
- Task set: 30 P14-E2/P14-B3 natural tasks
- Planner: Qwen3.5-9B Base, existing action budget
- Corrective RAG: off
- Verifier shadow: off
- Labels: provisional; never modified to fit outputs
- Original-BGE trace asset:
  `runs/advanced_retrieval_v2/sft_trace_candidates_original_bge.jsonl` (30 rows)
  with `provisional_not_expert_gold`; P18 Reserved excluded

Known failure for SFT attention:

- `0051` is a selection failure, not a retrieval miss.
- Enriched-BGE `0045`/`0065` are selection failures; `0051` is the candidate-recall loss.
- The next training target should be **Planner selection/ranking over a fixed candidate
  pool**, not another retriever, until retrieval changes are shown to help end-to-end.

## 8. Engineering and Regression

- Original index and original BM25 remain available and switchable.
- Gate remains the hard-constraint authority; no candidate/Ledger stale-state change.
- UNKNOWN and Memory/ASK behavior unchanged.
- Corrective RAG and Verifier remain opt-in; neither was enabled in Agent A/B.
- P18 Frozen Manifest was not modified. The known P18 hash failure remains because
  `catalog_search.py` was intentionally changed in Phase 1; it was not silently fixed.
- Local targeted regression after Phase 2 changes: **149 passed**.
- Full suite was not claimed clean; a previous full-suite attempt exceeded the time
  budget and had untriaged failures.
- Phase 2 experiment driver commit: `443d605`. The final report/handoff documentation
  commit follows it; final HEAD is recorded by `git log -1` in the Phase 2 response.
- No Push was performed in this round.

## 9. GPU Cost Summary

| Pass | Time | VRAM observation |
|---|---:|---|
| Enriched BGE build | 857.7 s | ~2,626 MiB sampled |
| Qwen3 reranker | 99.2 s (load 20.1 + score 79.1) | 4,037 MiB allocated; ~9,774 MiB nvidia peak |
| Qwen Agent A/B | 508.9 + 534.8 s | 19,444 MiB peak / 4,640 MiB free |

No Qwen and reranker were resident simultaneously. Remote GPU is idle at handoff.
