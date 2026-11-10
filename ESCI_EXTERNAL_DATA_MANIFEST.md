# ESCI External Evaluation — Data Manifest

**Status:** frozen ESCI English-US Task 1 judged-candidate reranking evaluation  
**Official source:** `https://github.com/amazon-science/esci-data`  
**Pinned commit:** `7916cdf6ab75a462e77f20ab40428a10923998d5` (main branch at download time)  
**License:** Apache-2.0 (official repository `LICENSE`/README; dataset citation: Reddy et al., arXiv:2206.06588)  
**Official Task 1 filter:** `small_version == 1`, `split == "test"`, `product_locale == "us"`  
**Expected official counts (README Table 1):** 8,956 queries / 181,701 judgments / avg depth 20.29

## 1. Raw files and SHA-256

| File | Bytes | SHA-256 | Verification |
|---|---:|---|---|
| `shopping_queries_dataset_examples.parquet` | 51,286,808 | `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` | matches Git-LFS pointer OID |
| `shopping_queries_dataset_products.parquet` | 1,108,857,465 | `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` | matches Git-LFS pointer OID |
| `shopping_queries_dataset_sources.csv` | 1,682,802 | `a5fed8ecc016443de40bf3c63098f0e3f23bbe4daa4236f1c38b8c3184778c50` | official raw file |

Raw examples overall have 2,621,288 rows before the official Task 1 filters. The official README states the small-version US test set has 8,956 queries and 181,701 judgments; our filtered data reproduces those exact counts.

## 2. Frozen filtered dataset

Local files:

- `runs/esci_external/esci_us_test_candidates.jsonl` — 181,701 candidate rows, one row per official judged `<query, product>` pair.
- `runs/esci_external/esci_us_test_queries.jsonl` — 8,956 query records.
- `runs/esci_external/frozen_esci_manifest.json` — machine-readable source/config/count manifest.

| Audit item | Result |
|---|---:|
| Queries | **8,956** |
| Candidate/judgment rows | **181,701** |
| Unique query IDs | 8,956 |
| Unique candidate product IDs | 164,900 |
| Duplicate `<query, product>` pairs removed | 0 |
| Queries without candidates | 0 |
| Queries with all product metadata | 8,956 / 8,956 |
| Missing product metadata candidates | 0 |
| Empty product titles | 0 |
| Average candidate depth | 20.29 |
| Minimum / maximum depth | 8 / 95 |
| Labels | E=79,708; S=63,563; C=8,099; I=30,331 |
| Queries with at least one E/S/C | 8,956 / 8,956 |

The observed maximum depth of 95 is the actual official small-version test data; the README's “up to 40” is not a hard row-count guarantee in this release. No candidate was dropped or added.

## 3. Candidate-set semantics

This is **judged-candidate reranking**, not full-catalogue retrieval:

- For every query, the candidate set is exactly the official judged product list.
- **BM25** and **BGE** rank the same candidate set; no unjudged products were added.
- No target/label is passed to either scorer; labels are used only for nDCG computation.
- No full-catalogue Recall or full-catalogue retrieval claim is made.
- The ESCI product set is external and separate from AgentRec-X's 76,263 Electronics catalogue; no mixing and no main-system BGE index modification.

## 4. Fixed scoring configuration

| Component | Frozen choice |
|---|---|
| Query text | official ESCI query text, unchanged |
| Product text | `product_title` only; same field for BM25 and BGE |
| BM25 | same tokenizer/constants as AgentRec-X (`k1=1.2`, `b=0.75`, lowercase alphanumeric runs, no stemming/stopwords), but document statistics are computed **within each query's judged candidate set** (candidate-local BM25) |
| BM25 tie-break | score desc, `product_id` asc |
| BGE | `BAAI/bge-base-en-v1.5`, original local weights; query prefix `Represent this sentence for searching relevant passages: `; CLS pooling; L2 normalization; max length 512; product titles without query prefix; score = normalized dot product |
| BGE tie-break | score desc, `product_id` asc |
| Semantic gain 口径（主结果） | E=1.0, S=0.1, C=0.01, I=0.0; official trec_eval **linear** nDCG: gain/log2(rank+1) |
| 官方脚本口径（sensitivity） | official script mapping E=1.0, S=0.01, C=0.1, I=0.0; reported separately, not mixed |
| nDCG positions | @10 and all judged candidates |
| Training / test tuning | none |
| Target injection | none |

### Official evaluation-script note

The official repository `ranking/prepare_trec_eval_files.py` maps `relevance_pos = {E:4, S:2, C:3, I:1}` and invokes `trec_eval` with `-m 'ndcg.1=0,2=0.01,3=0.1,4=1'`. That assigns **S=0.01 and C=0.1**, i.e. a different S/C ordering from the task-specified official nDCG gain `E=1, S=0.1, C=0.01, I=0`. The main result uses the semantic gain 口径; the official script mapping is reported separately as the official script 口径 sensitivity check. The official `trec_eval` nDCG implementation (`m_ndcg.c`) uses **linear** gain: `DCG = Σ gain/log2(rank+1)`, with ideal DCG from the same candidate labels. This is the primary formula; no exponential `(2^gain−1)` variant is used.

## 5. Reproduction

```bash
# raw files are in data/esci_external/ (large, git-ignored)
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python runs/esci_external/prepare_esci.py

# GPU encoding (existing AutoDL RTX 4090; no instance lifecycle operation)
scp -P 25405 -i ~/.ssh/agentrecx_autodl runs/esci_external/esci_gpu_inputs.tar.gz \
  runs/esci_external/encode_esci_gpu.py root@connect.cqa1.seetacloud.com:/root/esci_eval/
ssh -p 25405 -i ~/.ssh/agentrecx_autodl root@connect.cqa1.seetacloud.com \
  "cd /root/esci_eval && /root/autodl-tmp/bge_env/bin/python encode_esci_gpu.py \
   --product-jsonl runs/esci_external/esci_product_texts.jsonl \
   --query-jsonl runs/esci_external/esci_query_texts.jsonl \
   --model /root/autodl-tmp/models/bge-base-en-v1.5 --out-dir /root/esci_eval --batch-size 512"

# copy products.npz/queries.npz back, then evaluate locally
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python runs/esci_external/evaluate_esci.py
```

A local ESCI archive commit is created; no push/release/deploy is performed. The raw ESCI files and generated vectors are external/analysis artifacts, not part of the AgentRec-X main system.

## 8. CPU → GPU encoding migration (frozen inputs, no semantic changes)

### 8.1 Local CPU state

- Host: Windows `JKLOVE`, WSL Ubuntu-22.04; repo `/mnt/d/IT/CODE/PROJECTION/agentrec/X`.
- Two product shards were running under PIDs `170727` / `170734` (wrappers `170719` / `170728`).
- Stopped at approximately **2026-11-10 13:09 Asia/Shanghai**; progress was **35,840 / 82,450** product titles per shard.
- Completed before stop: **8,956 CPU query vectors** (`runs/esci_external/bge_query_vectors_cpu.npz`). No product vector file had been saved.
- Logs preserved: `runs/esci_external/encode_products_s0.log`, `encode_products_s1.log`.
- Only those exact ESCI encoder processes were stopped; no other Python tasks were terminated.

### 8.2 AutoDL RTX 4090 state

- SSH endpoint: `connect.cqa1.seetacloud.com:25405`; container `autodl-container-06fa4ab99e-6a73b008`.
- GPU: **NVIDIA GeForce RTX 4090, 24,564 MiB**; CUDA **12.8**; PyTorch **2.11.0+cu128**; `torch.cuda.is_available() == True`.
- BGE model: `/root/autodl-tmp/models/bge-base-en-v1.5`; weight SHA-256 `4b9d4a4b2d7ab06e3861861d5ee370974b31d17c0bb6af6f13d3dc79f33bbdcd` (matches local).
- Tokenizer `tokenizer.json`, `vocab.txt`, `tokenizer_config.json` SHA-256 match local exactly.
- No AutoDL instance lifecycle operation was performed; the existing instance was used as-is.

### 8.3 Encoding result

- Batch size **512**, no OOM fallback.
- Product titles: 164,900; time **80.464 s**; **2,049.36 items/s**.
- Queries: 8,956; time **1.673 s**; **5,352.83 items/s**.
- Total wall time **~82.1 s**; embedding dimension **768**.
- Sampled GPU utilization: avg **61.08%**, max **100%** (112 one-second samples; 83 nonzero, 78 ≥50%).
- Sampled hardware memory peak **11,439 MiB**; PyTorch allocated peak **4,476.6 MiB**.
- Products SHA-256: `60e9477774b8556dd5fd930eddfc219fde8e66769d134fb6c8eaf7189ae7006f`.
- Queries SHA-256: `4a9d17cab133d079faee86b9906490cc11d0e226bc93593960559eb1ba26b8cc`.
- All vectors finite; L2 norms ≈ 1.0; no missing/duplicate IDs.

### 8.4 CPU/GPU protocol consistency

- Query vectors all 8,956 CPU vs GPU: max abs **7.75e-7**, mean abs **7.17e-8**, min cosine **0.99999976**, mean cosine **1.0**.
- Product sample 128 CPU vs GPU: max abs **4.47e-7**, mean abs **6.08e-8**.
- Score sample 32 queries × 128 products: max abs **1.73e-6**, mean **1.04e-6**; top-1 **32/32**, top-5 overlap **1.0**.
- Evaluation uses **GPU vectors for both queries and products**; CPU vectors are reference-only. No CPU/GPU vectors are mixed.

### 8.5 Machine-readable manifests

- `runs/esci_external/gpu_encoding_manifest.json` — host/GPU/model/timing/hash/consistency record.
- `runs/esci_external/gpu_encoding_report.json` — encoder report including peak CUDA memory and throughput.
- `runs/esci_external/gpu_samples.csv` — GPU utilization/memory one-second samples.
- `runs/esci_external/cpu_gpu_query_diff.json` and `cpu_gpu_product_diff.json` — consistency checks.

