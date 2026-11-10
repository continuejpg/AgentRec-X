# ESCI External Evaluation — Reproducibility Archive

This directory contains the reproducible source and frozen results for the external ESCI
English-US Task 1 judged-candidate reranking evaluation. It is independent of AgentRec-X's
76,263-item Electronics catalogue and does not alter any main-system component.

## What is archived here

- `prepare_esci.py` — filters official `small_version=1`, `split=test`, `product_locale=us`, joins
  product metadata, deduplicates and records source hashes/counts.
- `export_gpu_inputs.py` — exports only unique query texts and unique product titles for GPU
  encoding (no labels sent to GPU).
- `encode_esci_gpu.py` — frozen BGE encoding protocol for CUDA, batch/OOM handling and hashes.
- `evaluate_esci.py` — candidate-local BM25, BGE scoring, official linear nDCG, paired bootstrap,
  query-type/failure analysis.
- `cpu_gpu_product_sample.py` — optional small CPU/GPU consistency check.
- `encode_esci_bge.py` — CPU/reference encoder retained for protocol comparison; not used for the
  final full embedding run.
- `reference/official_gain_audit.json` — pinned-commit audit of the official ESCI gain mapping.
- `results/` — all 8,956 per-query results and all 1,529 failure rows, compressed; metrics,
  failure samples, frozen manifest, GPU encoding evidence and CPU/GPU consistency JSON.
- `ARCHIVE_MANIFEST.json` — authoritative hashes, protocol, metric definitions, seed, counts.

Large artifacts are intentionally **not** committed: official raw parquet/CSV, all `.npz`
embeddings, candidate JSONL, raw logs. They can be regenerated from the commands below.

## Official data

- Repository: `https://github.com/amazon-science/esci-data`
- Pinned commit: `7916cdf6ab75a462e77f20ab40428a10923998d5`
- License: Apache-2.0; citation: Reddy et al., arXiv:2206.06588.
- Filter: `small_version == 1`, `split == "test"`, `product_locale == "us"`.
- Frozen test: **8,956 queries / 181,701 judged candidates**.

Raw files must be downloaded separately from the official Git-LFS paths into
`data/esci_external/` and verified against the SHA-256 values in `ARCHIVE_MANIFEST.json`.

## Two nDCG口径

The primary result is explicitly the **semantic gain口径**:

```text
E=1.0, S=0.1, C=0.01, I=0.0
```

using the official `trec_eval` **linear** nDCG formula:

```text
DCG = sum(gain / log2(rank + 1))
```

The pinned official evaluation script instead maps relevance positions
`E:4, S:2, C:3, I:1` and invokes trec_eval with gains `0, 0.01, 0.1, 1`, which is equivalent to
`E=1, S=0.01, C=0.1, I=0`. That result is archived separately as the **official script口径**
sensitivity result. The two口径 are never mixed.

Primary semantic-gain results (8,956 queries):

| Metric | BM25 | BGE-base |
|---|---:|---:|
| nDCG@10 | 0.690665 | 0.736566 |
| nDCG_all | 0.818397 | 0.844933 |

Official-script-gain sensitivity:

| Metric | BM25 | BGE-base |
|---|---:|---:|
| nDCG@10 | 0.669118 | 0.714406 |
| nDCG_all | 0.797493 | 0.824220 |

This is **judged-candidate reranking**, not full-catalogue retrieval. No full-catalogue Recall is
reported and no unjudged product is treated as irrelevant.

## Frozen BGE protocol

- Model: `BAAI/bge-base-en-v1.5`; original weights.
- Query prefix: `Represent this sentence for searching relevant passages: `
- Product text: `product_title` only, no prefix.
- CLS pooling; L2 normalization; max length 512; fp32; TF32 disabled.
- Batch size 512 on the RTX 4090 run; no OOM fallback.
- Query and product embeddings are encoded once per unique ID; no pair-level duplicate encoding.
- No training, fine-tuning, test tuning, or weight changes.

## Reproduction from a clean repo

From the repository root:

```bash
# 1) Obtain official raw data into data/esci_external/ and verify SHA-256 against ARCHIVE_MANIFEST.
# 2) Filter and freeze the official Task 1 test set:
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python experiments/esci_external/prepare_esci.py

# 3) Export unique query/product texts:
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python experiments/esci_external/export_gpu_inputs.py

# 4) GPU encoding (example AutoDL command; use the existing model path):
scp -P 25405 -i ~/.ssh/agentrecx_autodl runs/esci_external/esci_gpu_inputs.tar.gz \
  experiments/esci_external/encode_esci_gpu.py root@connect.cqa1.seetacloud.com:/root/esci_eval/
ssh -p 25405 -i ~/.ssh/agentrecx_autodl root@connect.cqa1.seetacloud.com \
  "cd /root/esci_eval && /root/autodl-tmp/bge_env/bin/python encode_esci_gpu.py \
   --product-jsonl runs/esci_external/esci_product_texts.jsonl \
   --query-jsonl runs/esci_external/esci_query_texts.jsonl \
   --model /root/autodl-tmp/models/bge-base-en-v1.5 --out-dir /root/esci_eval --batch-size 512"

# 5) Place products.npz/queries.npz in runs/esci_external as bge_product_vectors.npz /
#    bge_query_vectors.npz, then evaluate:
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python experiments/esci_external/evaluate_esci.py
```

`evaluate_esci.py` uses the frozen candidate list and labels only for nDCG; no label/target is
passed to either ranking method. The bootstrap seed is **20261110**.

## Re-verify the archive without GPU encoding

```bash
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python experiments/esci_external/verify_archive.py
```

This decompresses the archived per-query/failure results, checks counts and hashes, recomputes the
aggregate metrics from all 8,956 rows, checks the pinned official gain audit, and imports the
archived Python modules. It does **not** run GPU encoding or re-rank the Test set.

## Scope and non-goals

- Do not describe these numbers as full-catalogue retrieval, Recall, or a leaderboard result.
- Do not modify the frozen ESCI test or tune on it.
- Do not mix with AgentRec-X's 76,263-item catalogue, C4 proxy results, or Reserved Test.
- Do not modify Planner/V0/Gate/M9/MCP/Web/main BGE index.
- ESCI is an external product collection; its IDs are not AgentRec-X item IDs.

## License and citation

ESCI data: Apache-2.0, `amazon-science/esci-data`, commit `7916cdf6`; cite Reddy et al.,
*Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*,
arXiv:2206.06588.
