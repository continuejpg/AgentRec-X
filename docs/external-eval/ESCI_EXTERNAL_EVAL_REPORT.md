# ESCI Independent Product Search Evaluation

**Task:** official ESCI English-US Task 1, `small_version=1`, `split=test`, `product_locale=us`  
**Mode:** **judged-candidate reranking** (not full-catalogue retrieval)  
**Official test size:** **8,956 queries / 181,701 judged candidates**  
**Verdict:** **COMPLETED / PASS for the scoped external reranking evaluation.** BGE-base is better than BM25 on this frozen external test, but the result is not a pass condition and must not be described as full-catalogue Recall.

> The ESCI product collection is external and separate from AgentRec-X's 76,263 Electronics
> catalogue. No internal C4 proxy data was mixed in. No main-system Planner/V0/Gate/M9/MCP/Web
> or main BGE index was modified. No training, fine-tuning, test tuning, or leaderboard run was
> performed.

---

## 1. Official source and integrity

| Item | Value |
|---|---|
| Official repository | `https://github.com/amazon-science/esci-data` |
| Pinned commit | `7916cdf6ab75a462e77f20ab40428a10923998d5` |
| License | **Apache-2.0** (`LICENSE` and README of the official repository) |
| Paper | Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*, arXiv:2206.06588 |
| Task/filter | `small_version == 1`, `split == "test"`, `product_locale == "us"` |

Raw files downloaded from the official Git-LFS paths and verified against their LFS pointers:

| File | Bytes | SHA-256 |
|---|---:|---|
| `shopping_queries_dataset_examples.parquet` | 51,286,808 | `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` |
| `shopping_queries_dataset_products.parquet` | 1,108,857,465 | `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` |
| `shopping_queries_dataset_sources.csv` | 1,682,802 | `a5fed8ecc016443de40bf3c63098f0e3f23bbe4daa4236f1c38b8c3184778c50` |

The official README states the English-US Task 1 test set has **8,956 queries / 181,701 judgments**. Our filtered data reproduces these exact counts.

### Filtering result

| Audit item | Result |
|---|---:|
| Queries | **8,956** |
| Candidate rows / judgments | **181,701** |
| Unique query IDs | 8,956 |
| Unique candidate product IDs | 164,900 |
| Duplicate `<query, product>` pairs removed | 0 |
| Queries without candidates | 0 |
| Missing product metadata candidates | 0 |
| Queries with full product metadata | 8,956 / 8,956 |
| Empty product titles | 0 |
| Average depth | 20.29 |
| Minimum / maximum depth | 8 / 95 |
| Labels | E=79,708; S=63,563; C=8,099; I=30,331 |
| Queries with at least one E/S/C | 8,956 / 8,956 |

The observed depth maximum of 95 is the actual official small-version test data; the README's “up to 40” is not a hard row-count guarantee in this release. No judged candidate was dropped and no unjudged product was added.

Data/manifest files:

- `ESCI_EXTERNAL_DATA_MANIFEST.md`
- `runs/esci_external/frozen_esci_manifest.json`
- `runs/esci_external/gpu_encoding_manifest.json`
- `runs/esci_external/esci_us_test_candidates.jsonl` (181,701 rows)
- `runs/esci_external/esci_us_test_queries.jsonl` (8,956 rows)

---

## 2. Computation environment and CPU→GPU migration

### 2.1 Local CPU encoding (stopped)

| Item | Value |
|---|---|
| Host | Windows `JKLOVE`; WSL `Ubuntu-22.04` |
| Repository path | `/mnt/d/IT/CODE/PROJECTION/agentrec/X` |
| Encoder | two CPU product shards, PIDs `170727` / `170734`; wrapper PIDs `170719` / `170728` |
| Progress at stop | **35,840 / 82,450** product titles per shard |
| Completed before stop | **8,956 CPU query vectors**: `bge_query_vectors_cpu.npz` (preserved) |
| Product vectors saved | **none** (saved only at shard completion; stopping preserved partial progress only in logs) |
| Stop time | **2026-11-10 13:09 (Asia/Shanghai)** |
| Other Python tasks stopped | **none** |

Both encoder processes and their exact wrappers were terminated with `SIGTERM`; all logs and the completed CPU query cache were preserved. No product vector file was overwritten.

### 2.2 AutoDL RTX 4090 encoding

| Item | Value |
|---|---|
| SSH endpoint | `connect.cqa1.seetacloud.com:25405` |
| Container | `autodl-container-06fa4ab99e-6a73b008` |
| GPU | **NVIDIA GeForce RTX 4090, 24,564 MiB** |
| Actual CUDA | **True** (`torch.cuda.is_available() == True`) |
| CUDA / PyTorch | CUDA **12.8**; PyTorch **2.11.0+cu128** |
| Model | `/root/autodl-tmp/models/bge-base-en-v1.5`; weights SHA-256 `4b9d4a4b2d7ab06e3861861d5ee370974b31d17c0bb6af6f13d3dc79f33bbdcd` (same as local) |
| Tokenizer | `tokenizer.json`, `vocab.txt`, `tokenizer_config.json` hashes **match local exactly** |
| Batch size | **512**; no OOM fallback |
| Labels on GPU | **none**; only unique query/product texts were transferred |

GPU result:

| Measure | Value |
|---|---:|
| Product titles encoded | 164,900 |
| Queries encoded | 8,956 |
| Embedding dimension | 768 |
| Product encoding | **80.464 s**, **2,049.36 items/s** |
| Query encoding | **1.673 s**, **5,352.83 items/s** |
| Total encoding wall time | **~82.1 s** |
| Sampled GPU utilization | avg **61.08%**, max **100%** (112 one-second samples; 83 nonzero, 78 ≥50%) |
| Sampled GPU memory peak | **11,439 MiB** hardware-reported |
| PyTorch allocated peak | **4,476.6 MiB** |
| NaN / Inf | none; all finite |
| Product/query norms | 0.9999998–1.0000001 (L2) |
| Product vectors SHA-256 | `60e9477774b8556dd5fd930eddfc219fde8e66769d134fb6c8eaf7189ae7006f` |
| Query vectors SHA-256 | `4a9d17cab133d079faee86b9906490cc11d0e226bc93593960559eb1ba26b8cc` |

The two encoding runs produced identical vector hashes. The actual GPU was used; the remote process was not merely connected to the server and run on CPU.

### 2.3 CPU/GPU protocol consistency

Protocol was unchanged: BGE query prefix, title-only product text, CLS pooling, L2 normalization, max length 512, float32, no TF32.

| Check | Result |
|---|---|
| Query vectors, CPU vs GPU (all 8,956) | max abs **7.75e-7**, mean abs **7.17e-8**, min cosine **0.99999976**, mean cosine **1.0** |
| Product sample, CPU vs GPU (128 titles) | max abs **4.47e-7**, mean abs **6.08e-8**, min cosine **0.99999982** |
| CPU/GPU retrieval score sample (32 queries × 128 products) | max score diff **1.73e-6**, mean **1.04e-6** |
| Top-1 / Top-5 agreement | **32/32 top-1**, **1.0 mean top-5 overlap** |

The final evaluation uses **GPU vectors for both queries and products**; the CPU query vectors are reference-only. CPU/GPU differences are normal floating-point/transformer-kernel differences and are far below ranking-changing magnitude.

---

## 3. Frozen evaluation protocol

### 3.1 Candidate set

- For each query, the candidate set is exactly the **official judged candidate list**.
- BM25 and BGE rank the **same candidate set**.
- No judged product was removed and no unjudged product was added.
- This is **judged-candidate reranking**. It is **not** a full-catalogue retrieval task, and no full-catalogue Recall is reported.

### 3.2 Fixed scoring configuration

| Component | Frozen value |
|---|---|
| Query text | official ESCI query, unchanged |
| Product text | `product_title` only; same field for BM25 and BGE |
| BM25 | AgentRec-X tokenizer/constants: lowercase alphanumeric runs, no stemming/stopwords, `k1=1.2`, `b=0.75`; document statistics are **candidate-local** (within each query's judged candidate set) |
| BM25 tie-break | score desc, `product_id` asc |
| BGE | `BAAI/bge-base-en-v1.5`; original weights; query prefix `Represent this sentence for searching relevant passages: `; product title without prefix; CLS pooling; L2 normalization; max length 512; score = normalized dot product |
| BGE tie-break | score desc, `product_id` asc |
| Training / test tuning | none |
| Target/label injection into scoring | none |

The BM25 candidate-local construction is a reranker baseline over the judged list; it is not the main-system full-catalogue BM25 and is not compared to internal 76,263-catalogue numbers.

### 3.3 Official nDCG formula

The official `trec_eval` nDCG implementation (`m_ndcg.c`) uses **linear gain**:

```text
DCG  = Σ gain / log2(rank + 1)
IDCG = ideal ordering of the same candidate gains
nDCG = DCG / IDCG
```

主结果使用 **semantic gain 口径**（E=1.0, S=0.1, C=0.01, I=0.0）。  
Reported metrics: **nDCG@10** and **nDCG over all judged candidates** (`nDCG_all`). Mean is the query-level macro mean over 8,956 queries.

### 3.4 官方脚本口径 audit / sensitivity

The official repository's `ranking/prepare_trec_eval_files.py` maps
`relevance_pos = {E:4, S:2, C:3, I:1}` and invokes trec_eval with
`-m 'ndcg.1=0,2=0.01,3=0.1,4=1'`. That assigns **S=0.01, C=0.1**, a different
S/C ordering from the semantic gain 口径 **S=0.1, C=0.01**. The main result uses the
semantic gain 口径; the official script mapping is reported separately as the
**official script 口径** sensitivity result and is not mixed with the main result.

---

## 4. Results: BM25 vs BGE

### 4.1 主结果：semantic gain 口径（official linear nDCG，n = 8,956）

| Metric | BM25 | BGE-base | BGE − BM25 |
|---|---:|---:|---:|
| **nDCG@10** | **0.690665** | **0.736566** | **+0.045901** |
| 95% CI (query bootstrap) | [0.686335, 0.695004] | [0.732499, 0.740692] | **[0.041966, 0.049705]** |
| **nDCG_all** | **0.818397** | **0.844933** | **+0.026536** |
| 95% CI | [0.815213, 0.821507] | [0.841912, 0.847941] | **[0.023928, 0.029191]** |

Paired query statistics:

| Metric | BGE wins | BM25 wins | Ties |
|---|---:|---:|---:|
| nDCG@10 | 5,230 | 3,381 | 345 |
| nDCG_all | 5,278 | 3,507 | 171 |

BGE is better on both aggregated metrics, with paired CIs excluding zero. This is an
external benchmark result, not a leaderboard submission and not a full-catalogue retrieval
claim.

### 4.2 官方脚本口径（sensitivity）

Using the official repo script's S/C mapping (still linear trec_eval nDCG):

| Metric | BM25 | BGE-base | BGE − BM25 | Paired 95% CI |
|---|---:|---:|---:|---:|
| nDCG@10 | 0.669118 | 0.714406 | +0.045288 | [0.041043, 0.049489] |
| nDCG_all | 0.797493 | 0.824220 | +0.026727 | [0.023822, 0.029720] |

The direction and approximate magnitude are stable under the **official script 口径** sensitivity mapping.

---

## 5. Query-type analysis

All query-type flags are **deterministic lexical/heuristic** signals, not manually verified semantic categories. Flags can overlap. Groups with small n are descriptive only.

| Flag | n | BM25 nDCG@10 | BGE nDCG@10 | BGE − BM25 | BM25 nDCG_all | BGE nDCG_all |
|---|---:|---:|---:|---:|---:|---:|
| negation | 731 | 0.562338 | 0.585648 | +0.023310 | 0.764444 | 0.775320 |
| brand | 1,626 | 0.704307 | 0.752237 | +0.047930 | 0.827289 | 0.857574 |
| model-token | 693 | 0.670789 | 0.706330 | +0.035541 | 0.794764 | 0.819127 |
| attribute | 1,168 | 0.700490 | 0.742588 | +0.042099 | 0.825020 | 0.848881 |
| accessory | 595 | 0.697846 | 0.740139 | +0.042293 | 0.821724 | 0.849492 |
| main product | 8,361 | 0.690154 | 0.736311 | +0.046157 | 0.818160 | 0.844608 |

Definition basis:

- **negation**: regex for `not|no|without|except|excluding|avoid|non`.
- **brand**: query contains a candidate product brand string (candidate metadata exact phrase).
- **model-token**: query has an alphanumeric token containing both letters and digits.
- **attribute**: query contains a fixed feature/attribute lexicon.
- **accessory**: query contains accessory nouns such as case/cover/charger/cable/adapter/mount/holder.
- **main product**: complement of the accessory flag.

These are not ground-truth semantic labels and should not be used for causal claims. BGE is numerically higher in every flag group, with the smallest difference on negation (+0.0233).

---

## 6. Failure analysis

All 8,956 per-query records are retained in `esci_per_query.jsonl`; no failure was deleted.
A failure subset is also written for convenience:

- **Both methods at nDCG@10 = 0**: **7 queries**
- **BGE material loss** (BGE < BM25 by at least 0.10 nDCG@10): **1,522 queries**
- Failure rows written: **1,529**
- BGE material wins (BGE > BM25 by at least 0.10): **2,960 queries**
- BGE any-win / BM25 any-win / tie: 5,230 / 3,381 / 345

### Representative BGE material losses (actual query/product evidence)

1. **`cryptid`** (query_id 30411; 16 candidates; E=1, I=15). BM25 nDCG@10 **1.0**, BGE **0.0**. BM25 top is labeled E (`The Awakening`); BGE top items are cryptid books/products labeled I. The query is a single ambiguous word and the observed Exact product is semantically surprising; this is both a query/context ambiguity and a possible label-semantics issue.
2. **`miller specteum 375`** (query_id 69143; 16 candidates; E=1, S=2, C=13). BM25 **0.9604**, BGE **0.0681**. BM25 top is `Miller Electric 907529 Plasma Cutter, Inverter, Spectrum 375` (E); BGE ranks complement/accessory tips and nozzles ahead of the Exact cutter. Exact brand/model match favors BM25; BGE is pulled toward accessory semantics.
3. **`.625 width 20 gal metal strapping without holes`** (query_id 358; E=1, S=3, I=12). BM25 **0.9620**, BGE **0.0854**. BM25 top is an Exact strapping tensioner/cutter; BGE top is an Irrelevant hanger strap, followed by a Substitute. The negation/specification “without holes” and the exact product form are not handled well by the dense representation.
4. **`mens professional make-up kit`** (query_id 67807; E=1, S=4, C=11). BM25 **0.8816**, BGE **0.0943**. BM25 top is an Exact hair-cutting-scissors set (surprising but labeled E); BGE returns toiletry/makeup bags (C). The query is ambiguous and the observed Exact product is not the most obvious semantic match.
5. **`toddler play makeup brushes`** (query_id 103728; E=2, S=14). BM25 **1.0**, BGE **0.2364**. BM25 ranks both E makeup-brush products first; BGE prefers Substitute pretend-play makeup sets. This is a graded-relevance/type-specificity case: BGE's semantic neighbors are close but not the Exact brush products.
6. **`soundstrem`** (query_id 96332; E=1, S=6, C=1, I=8). BM25 **0.8781**, BGE **0.1148**. BM25 finds `Soundstream BX-20Z Digital Bass Reconstruction Processor` (E); BGE picks an Irrelevant sound-deadening product and a Complement. The query is a misspelled brand; lexical exactness matters.

### Representative both-zero cases

- **`entertainment stand for tv up to 90`** (query_id 37789): top items are TV stands for up to 43/50/70/80 inches; the observed E target is not in the top 10 for either method. This is a long-tail/spec coverage failure, not a claim that the target is irrelevant.
- **`family costumes`** (query_id 39509): generic family intent, candidate list contains individual costumes; both methods return I items first, E is outside top 10.
- **`gixtri`** (query_id 45773): nonsensical/unknown query, top results are unrelated supplements/vitamins; both methods fail. This is query ambiguity and catalogue coverage.
- **`science auburn`** (query_id 90978): ambiguous phrase; both methods rank unrelated books/Auburn merchandise ahead of E.
- **`silk bodysuit long sleevetops`** (query_id 93331): candidate list contains long-sleeve tops but the Exact target is not top-10; query/attribute coverage.
- **`slim whitman cd box`** (query_id 94437): top results are slim CD storage cases, not the observed E; both methods fail on the product/spec relationship.

These examples show four recurring patterns: **exact model/brand/specification mismatch**, **negation/attribute specificity**, **graded relevance vs semantic neighbor**, and **ambiguous/long-tail query coverage**. The single-observed-positive labeling means a miss is not proof that all other candidates are irrelevant; all non-target products remain unjudged.

Artifacts:

- `runs/esci_external/esci_per_query.jsonl` — all 8,956 query rankings/metrics
- `runs/esci_external/esci_failures.jsonl` — 1,529 both-zero/material-loss rows
- `runs/esci_external/esci_failure_samples.json` — deterministic samples
- `runs/esci_external/esci_metrics.json` — aggregate metrics/CIs

---

## 7. Scope boundaries and license

- **Judged-candidate reranking only.** No full-catalog retrieval, no full-catalog Recall, no “retrieved all products” claim.
- **No unjudged products were treated as irrelevant**; only official judged candidates are used, and metrics reflect ESCI judgments.
- **External data:** ESCI product set is separate from AgentRec-X's 76,263 Electronics catalogue. No internal C4 proxy test data was mixed in.
- **No main-system change:** no Planner, V0, Gate, M9, MCP, Web, main BGE index, or retrieval weights were changed.
- **No training/fine-tuning/test tuning**: BGE original weights; no ESCI parameter selection; no test reranking after seeing results.
- **License:** official `amazon-science/esci-data` repository is Apache-2.0. Cite Reddy et al., arXiv:2206.06588. Raw files and vectors are kept as local external artifacts, not committed or redistributed in Git.
- **Pretraining caveat:** BGE base is a public pretrained model; overlap between its pretraining data and ESCI product/query text was not audited. This is an external evaluation of a frozen model, not a guarantee of zero pretraining exposure.

---

## 8. Direct answers

1. **Where did the previous CPU encoding run?**
   On local Windows host `JKLOVE` in WSL Ubuntu-22.04, repository `/mnt/d/IT/CODE/PROJECTION/agentrec/X`. Two product shards (PIDs 170727/170734) were running; 8,956 CPU query vectors were already complete.
2. **Was local long-running CPU encoding stopped?**
   Yes. The two exact shard PIDs and wrappers were stopped at around 13:09 Asia/Shanghai; logs and CPU query vectors were preserved; no product vectors had been saved. No other Python tasks were terminated.
3. **Did AutoDL GPU actually participate?**
   Yes. AutoDL RTX 4090 (24,564 MiB), CUDA 12.8, PyTorch 2.11.0+cu128, `torch.cuda.is_available() == True`; actual encoding ran on CUDA, not CPU. Sampled utilization averaged 61.08% and peaked at 100%.
4. **What were the GPU encoding time and throughput?**
   Products: **80.464 s / 2,049.36 items/s**; queries: **1.673 s / 5,352.83 items/s**; total ~**82.1 s**; batch 512; no OOM; 164,900 products and 8,956 queries; 768-d; peak hardware memory **11,439 MiB**, torch allocated peak **4,476.6 MiB**.
5. **Are CPU and GPU protocols consistent?**
   Yes, within normal float error. Tokenizer/weights are identical; query vectors max abs **7.75e-7** and mean cosine **1.0**; product-sample max abs **4.47e-7**; score max abs **1.73e-6**; top-1 **32/32** and top-5 overlap **1.0**. Evaluation uses GPU vectors for both sides; CPU vectors are reference-only.
6. **Was the ESCI evaluation completed?**
   Yes. Official filters produced **8,956 queries / 181,701 judged candidates**; BM25 and BGE ranked identical candidate sets under the frozen protocol; all per-query results and failure cases are retained.
7. **What are the official metrics and confidence intervals?**
   Semantic gain 口径（主结果）nDCG@10: BM25 **0.690665** [0.686335, 0.695004], BGE **0.736566** [0.732499, 0.740692]; paired difference **+0.045901** [0.041966, 0.049705]. nDCG_all: BM25 **0.818397** [0.815213, 0.821507], BGE **0.844933** [0.841912, 0.847941]; paired **+0.026536** [0.023928, 0.029191]. BGE wins/losses/ties at nDCG@10: 5,230 / 3,381 / 345.
8. **Are data and scripts still reproducible?**
   Yes: official source/commit/license and SHA-256 are recorded; `prepare_esci.py`, `export_gpu_inputs.py`, `encode_esci_gpu.py`, `evaluate_esci.py`, and the GPU manifest/logs are present; input and output hashes are recorded. GPU vectors can be regenerated from the same remote model, tokenizer, prefix, pooling, normalization and max length.

---

## 9. Status

| Item | Status |
|---|---|
| Official ESCI source/license/commit verified | **PASS** |
| 8,956 queries / 181,701 judged candidates | **PASS** |
| Frozen test unchanged / no test tuning | **PASS** |
| Shared judged candidate set | **PASS** |
| BGE weights/tokenizer unchanged | **PASS** |
| Local CPU long encoding stopped | **PASS** |
| AutoDL RTX 4090 actually used | **PASS** |
| GPU embedding completeness/finiteness | **PASS** |
| CPU/GPU consistency | **PASS** (float-level) |
| Official linear nDCG + CIs | **PASS** |
| All failure cases retained | **PASS** (8,956 per-query; 1,529 failure subset) |
| Judged-candidate reranking scope stated | **PASS** |
| Full-catalogue retrieval | **NOT EXECUTED** |
| Main-system changes | **NONE** |
| Commit/push/deploy | **NONE** |
| **External evaluation** | **COMPLETED** |

## 10. Accurate resume wording

> Evaluated frozen BM25 and `BAAI/bge-base-en-v1.5` on the official ESCI English-US Task 1
> small-version test set (**8,956 real shopping queries, 181,701 judged candidates**), using
> judged-candidate reranking and official linear-gain nDCG (@10 and all candidates).
> BGE-base achieved nDCG@10 **0.7366** vs BM25 **0.6907** (paired difference **+0.0459**,
> 95% CI **+0.0420 to +0.0497**). This is an external candidate-reranking benchmark, not a
> full-catalogue retrieval/Recall result.


## 11. Reproducibility archive

The archive source and all 8,956 per-query / 1,529 failure records are committed under
`experiments/esci_external/`:

- `experiments/esci_external/*.py` — six source scripts (five core + CPU protocol reference).
- `experiments/esci_external/results/esci_per_query.jsonl.gz` — all 8,956 query results.
- `experiments/esci_external/results/esci_failures.jsonl.gz` — all 1,529 both-zero/material-loss rows.
- `experiments/esci_external/results/esci_metrics.json` — aggregate metrics, CIs and gain labels.
- `experiments/esci_external/results/esci_failure_samples.json` — deterministic failure samples.
- `experiments/esci_external/ARCHIVE_MANIFEST.json` — source/result hashes, protocol, metric definitions, seed.
- `experiments/esci_external/reference/official_gain_audit.json` — pinned-commit ESCI gain audit.
- `experiments/esci_external/README.md` — clean-repo reproduction and scope.

`verify_archive.py` checks the archived gzip counts, hashes, recomputed aggregate metrics, the
official gain audit and imports without running GPU encoding. Raw official data/embeddings/logs
remain git-ignored and are regenerated from the commands in the README. Large assets were not
committed. A local archive commit is created; no Push/Release/Deploy is performed.

## 12. Reproduction

```bash
# 1) official raw data (large; git-ignored)
#    examples/products/sources from amazon-science/esci-data at commit 7916cdf6
# 2) filter/hash/freeze
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python runs/esci_external/prepare_esci.py
# 3) unique texts only
wsl ... -- .venv/bin/python runs/esci_external/export_gpu_inputs.py
# 4) GPU encoding on AutoDL (existing instance; no lifecycle operation)
scp -P 25405 -i ~/.ssh/agentrecx_autodl runs/esci_external/esci_gpu_inputs.tar.gz \
  runs/esci_external/encode_esci_gpu.py root@connect.cqa1.seetacloud.com:/root/esci_eval/
ssh -p 25405 -i ~/.ssh/agentrecx_autodl root@connect.cqa1.seetacloud.com \
  "cd /root/esci_eval && /root/autodl-tmp/bge_env/bin/python encode_esci_gpu.py \
   --product-jsonl runs/esci_external/esci_product_texts.jsonl \
   --query-jsonl runs/esci_external/esci_query_texts.jsonl \
   --model /root/autodl-tmp/models/bge-base-en-v1.5 --out-dir /root/esci_eval --batch-size 512"
# 5) copy products.npz/queries.npz back and evaluate
wsl ... -- .venv/bin/python runs/esci_external/evaluate_esci.py
# 6) optional CPU/GPU protocol sample
wsl ... -- .venv/bin/python runs/esci_external/cpu_gpu_product_sample.py
```

No AgentRec-X source, main BGE index, frozen ESCI test, or AutoDL instance lifecycle was modified.
