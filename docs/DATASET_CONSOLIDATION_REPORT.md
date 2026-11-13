# Dataset Consolidation & Gold-Supervised Recommendation Training — Phase A Report

## 0. Scope and outcome

This round is local/CPU data preparation. No GPU training was started. The AutoDL host is
reachable in no-GPU mode (11G free data disk; `nvidia-smi` unavailable by design).

Outcome:

- Complete inventory of the repository's P10–P20, retrieval, SFT, adapter, ranking and
  external-data assets.
- Dataset A built from **human-labeled Amazon ESCI**.
- Dataset B built from **real Amazon Reviews 2023 Electronics ratings** at exact
  `parent_asin` intersection with the current catalogue.
- Independent ESCI and TREC benchmarks prepared.
- Gold-label intersection with the current 76,918-item catalogue is **too low for direct
  Agent ranking fusion**.
- No new ranking fine-tuning is recommended for the current Agent. The old Planner SFT
  adapters are archived but are not candidates for ranking training.

Detailed machine-readable inventory: `docs/DATASET_ASSET_INVENTORY.json`.

## 1. Complete data asset inventory

Human-readable inventory: [DATASET_ASSET_INVENTORY.md](DATASET_ASSET_INVENTORY.md).

The inventory records 51 assets across A–F:

- **A Active runtime:** Electronics catalogue, metadata sidecar, enriched retrieval docs,
  original BGE index, Qwen3.5-9B base, BGE model, remote training environment, current
  project copy.
- **B Training candidates:** ESCI Dataset A, Amazon Reviews Dataset B, Planner SFT v0/v1,
  preference-operation SFT.
- **C Evaluation/frozen:** P14-D1/D2, P14-E2/B3, P15 evidence, P18 frozen/reserved, P20 task
  suite and arm results, Preference Ranking v1, TREC qrels.
- **D Reproducibility archives:** raw ESCI/TREC/review files, old P10–P14 experiment assets,
  SASRec/Two-Tower/TIGER/LR legacy assets, final Planner adapters.
- **E Regenerable temporary:** embedding caches, scratch pools, old index builds.
- **F Duplicate/obsolete:** ESCI CPU/GPU vector caches; old SFT checkpoints were already
  deleted in the preceding cleanup round.

Key protected assets:

```text
data/p13r/electronics/catalog_meta.jsonl              76,918 items
data/enrichment/electronics_metadata_sidecar.jsonl    76,263 items
runs/p14d2/bge_index.npz                              76,263 vectors
runs/p18/freeze_manifest.json
runs/p20_agent_eval/task_suite.json                   30 tasks
remote Qwen3.5-9B-ms
remote agentrecx-sft environment
remote /root/autodl-tmp/AgentRec-X-phase2
```

No P18 file was modified. No P18 Reserved task was used for tuning.

## 2. Deletion/archive and space

### This round

- **No destructive deletion.**
- Final Planner adapters were archived locally under `runs/dataset_archive/`:
  - Planner SFT v0 adapter: **185MB**;
  - Planner SFT v1 QLoRA adapter: **185MB**;
  - total archive 14 files / **386,382,808 bytes (~368MiB)**.
- No model weights were overwritten or quantised.

### Prior cleanup context

Old SFT Checkpoints had already been deleted in the preceding cleanup round:

```text
agentrecx/training/agent_sft/output/qwen3_5_9b_lora/checkpoints
agentrecx/training/agent_sft/output/qwen3_5_9b_lora_v2/checkpoints
agentrecx/training/planner_sft/output/qwen3_5_9b_planner_lora_v0/checkpoints
```

Approximate reclaim: 2.9GB; final adapters were preserved.

### Cleanup plan (not executed)

- Keep all active runtime, frozen, and final adapter assets.
- Do not delete raw ESCI/TREC/Amazon files; they are class D and not backed up elsewhere.
- ESCI duplicate caches are class F but are **not deleted**:
  - `runs/esci_external/bge_product_vectors_gpu.npz` duplicates
    `bge_product_vectors.npz` (470MB duplicate);
  - `bge_query_vectors_gpu.npz` duplicates `bge_query_vectors.npz` (25MB);
  - `bge_queries_shard0.npz` duplicates `bge_query_vectors_cpu.npz` (25MB).
  These are regenerable, but no deletion authorisation was given for them here.
- Keep final adapters:
  - `runs/dataset_archive/planner_sft_v0` SHA
    `54797dc5fc8d0aa67bd47facac6732f1a9886f08a289701e72f73572a84e96d8`;
  - `runs/dataset_archive/planner_sft_v1` SHA
    `3ed5d5ffbcd41481a4312765a227bf084a78646925cd1bc7cb1d10ca7a2ef222`;
  - `runs/p14c1/sft_adapter` SHA
    `f191c811cab6f1f3da98466c0c0341ccfd0bfa4105b371e1c1af4500631326b8`.

## 3. Current catalogue / index identity

```text
catalog_meta.jsonl rows/items: 76,918
BGE index documents:           76,263
accepted catalogue identity:   76,263
missing from BGE index:        655 catalogue item_ids
```

The 655 difference is the accepted-index filter used when the original BGE index was built;
the BGE manifest asserts identity equality over the accepted 76,263 ids and zero duplicates.
This is not a new retrieval change. Any future ranking index must use the same accepted
identity set and must not mix source/title matches.

## 4. ESCI human labels and Dataset A

Source:

- [amazon-science/esci-data](https://github.com/amazon-science/esci-data)
- official repository commit `7916cdf6ab75a462e77f20ab40428a10923998d5`
- license **Apache-2.0**
- task: small_version=1, product_locale=us, query-product relevance
- Human labels retained: **E / S / C / I**.

License boundary: ESCI is Apache-2.0. Reuse/redistribution of the resulting Dataset A must
retain attribution and the source terms; the current Agent catalogue is a separate asset.

Raw:

```text
examples 2,621,288 rows
products 1,814,924 rows
US small_version=1 rows: 601,354
US small_version=1 queries: 29,844
US small_version=1 products: 482,105
```

Dataset A: `data/gold/esci_relevance_v1/`

| Split | Rows | Queries | E | S | C | I |
|---|---:|---:|---:|---:|---:|---:|
| train | 378,413 | 18,847 | 164,473 | 132,424 | 17,302 | 64,214 |
| val | 41,240 | 2,041 | 17,346 | 15,204 | 1,788 | 6,902 |
| test | 181,701 | 8,956 | 79,708 | 63,563 | 8,099 | 30,331 |

Split rules:

- **official test preserved**;
- validation is a deterministic 10% of official train query groups;
- no query group crosses train/val/test;
- no product metadata passed to label mapping; product text joined after labels;
- no target/product title fuzzy matching.

Label mapping:

```text
raw labels: E exact, S substitute, C complement, I irrelevant (human)
primary gains: E=1.0, S=0.1, C=0.01, I=0.0
alternative official-script gains: E=1.0, S=0.01, C=0.1, I=0.0
```

Complement is **not** declared universally worse than Substitute; it is a different
relationship. Both mappings are stored; primary and sensitivity must not be mixed.

Dataset A SHA256:

```text
train 5b287f4faf3e83e9eb0a7f65d53c8b8fea93c80010ffd4aef4671def67282a30
val   138c5f73bfcb71cd45fd4799b6d37d75dc90d6859e7a874508205c3ed7f836c7
test  6cfb94b91172dc5ac9b1dcc88af9fe504e5c21dde150d2d38e122d726b0b9a96
```

Independent ESCI relevance benchmark (official test, judged-candidate reranking, prior real
BGE/BM25 external experiment):

```text
BM25 nDCG@10 0.690665; BGE nDCG@10 0.736566
BM25 nDCG_all 0.818397; BGE nDCG_all 0.844933
BM25 MRR@10 (binary E/S) 0.904570; BGE MRR@10 0.936004
BM25 MRR@10 (E/S/C) 0.937021; BGE MRR@10 0.954548
```

This is a judged-candidate benchmark, not full-catalogue retrieval. Test never used for
selection.

## 5. Amazon Reviews 2023 Electronics labels and Dataset B

Existing real rating-only files:

```text
/root/autodl-tmp/p13r/raw/Electronics.valid.csv
/root/autodl-tmp/p13r/raw/Electronics.test.csv
fields: user_id,parent_asin,rating,timestamp
```

These are **real user ratings**, not human pairwise preference labels. The files contain no
review text and are not an official Amazon train/val/test recommendation split.

License boundary: the Hugging Face card for `McAuley-Lab/Amazon-Reviews-2023` does not
declare a single license in its metadata; the upstream project/terms and any redistribution
must be checked before releasing a copy. The local derived Dataset B is kept as a research
artifact, not a redistributable benchmark. No raw review text is present.

Full scanned data:

```text
rows:            8,029,455
unique users:    5,139,179
unique items:      552,872
rating counts:   1: 1,286,028; 2: 467,218; 3: 546,299;
                  4: 819,214; 5: 4,910,694; 0: 2
users >=2:       1,319,622
users >=3:         537,732
users >=5:         164,989
users >=10:         33,828
users >=20:          7,814
```

Catalog intersection by exact `parent_asin` only:

```text
catalog items:             76,918
intersected items:         27,403
intersected rows:       1,104,764
intersected users:      1,005,308
catalog item coverage:      35.63%
catalog-row coverage:       13.76%
```

Dataset B: `data/gold/amazon_reviews_electronics_v1/`

Temporal per-user construction:

- history = earlier catalog interactions only;
- last target -> test;
- second-last -> val;
- earlier targets -> train;
- target is never placed in its own history;
- candidate universe = current full 76,918-item catalogue;
- all non-target items are **UNJUDGED**, never negative.

| Split | Examples |
|---|---:|
| train | 8,270 |
| val | 14,157 |
| test | 14,157 |

Other facts:

```text
users with >=3 catalog interactions: 14,157
users with 1 catalog interaction:     928,279 (excluded cold start)
users with 2:                         62,872
mean history length:                  2.204
max history length:                   60
test target ratings:
  5: 10,028; 4: 1,925; 3: 867; 2: 454; 1: 883
```

Files:

```text
train 7f791f964150a12bc178114775de7ea3eb82978867c6878f27d486debbb40fa4
val   3e2ee99e510beacbded5b070a3252e672882eaa48bccfbab950f983b3ad76a2b
test  c2dd2865d3e8b7fff9b9dc0c03ffff4e9d1155a55ea2de134f512709587dfded
```

External personalized benchmark, non-personalized popularity reference only:

```text
test tasks: 14,157
Recall@10:   0.141485
Recall@50:   0.276612
MRR@10:      0.045925
nDCG@10:     0.068027
target popularity coverage: 91.56%
```

These are **known-positive retrieval** numbers, not full relevance recall: the other 76,917
catalogue items are unjudged. No negative sampling was performed.

## 6. TREC 2025 independent benchmark

Source: [TREC 2025 Product Search and Recommendation](https://trec.nist.gov/data/product2025.html),
HF dataset `trec-product-search/product-recommendation-2025` (Apache-2.0 dataset card,
derived from ESCI/M2).

Downloaded:

```text
qrels.rec rows: 7,633
judged queries: 47
test queries: 50
queries without qrels: PSRT_Recs_012, PSRT_Recs_033, PSRT_Recs_046
qrels products: 7,103
current catalog exact intersection: 115 products (1.619%)
search qrels: 23,409 rows / 93 queries / 19,470 products
corpus: 17,652,317 bytes zstd; product text JSONL
```

Raw TREC relationship labels retained:

```text
S2 full substitute, S1 conceptual substitute, S0 related-not-substitute
C2 full complement, C1 complement-unknown-compatibility, C0 complementary-but-incompatible
NR not related, UA unable to assess
```

TREC is **test-only** and was not mixed into training; no synthetic negatives and no use of
current scorer outputs as labels.

## 7. Label/scale/ID conclusions

Exact ID alignment only; no title fuzzy matching:

- ESCI `product_id` ↔ current `item_id`: **626 products / 944 rows** in US small_version=1
  (1,673 products if all ESCI locales are included). These are external ASINs, not the
  current C4 catalogue products in general.
- TREC qrels ASIN ↔ current `item_id`: **115 / 7,103** (1.62%); TREC search 0 / 19,470.
- Amazon Reviews `parent_asin` ↔ current `item_id`: **27,403 items, 1,104,764 rows,
  1,005,308 users**.
- ESCI product metadata has 12,152 duplicate product_id rows; Dataset A uses exact
  `product_id+locale` and the first metadata row by deterministic dedupe.
- Amazon Reviews used exact `parent_asin`; no variant or title merge.
- Current catalogue includes 76,918 ids; the accepted dense index uses 76,263. The 655
  difference is the existing accepted-index filter, not a new match.

## 8. Is Ranking Training justified?

### Option 1 — ESCI generic reranker

**Data-feasible as an external research task**, but not for direct Agent fusion.

- 378k train rows / 18,847 queries; human E/S/C/I labels; official test preserved.
- Candidate groups are query-level; cross-encoder/listwise training is possible.
- Current Agent catalogue overlap is only 626 ESCI products, so a model trained here would
  not directly rerank the Agent's 76,263-item catalogue.
- No GPU run was started this round.

### Option 2 — Amazon personalized ranking

**Not justified.**

- Only 14,157 users have ≥3 catalog interactions; 928,279 users have exactly one.
- 8,270 pointwise training examples; mean history 2.204.
- No human pairwise preference labels; no negative candidate labels; unobserved items are
  unjudged.
- Training a ranker on these pointwise ratings would not establish ranking quality on the
  Agent's user requests.

### Option 3 — ESCI pretrain + Amazon personalization

**Not yet.** The personalized signal is too sparse and lacks reliable candidate negatives.

### Option 4 — independent benchmarks only

**Recommended for this phase.** Keep ESCI and TREC as independent external benchmarks; do
not fuse them into the Agent catalogue by title matching or partial IDs.

### Decision

```text
Do not train a new Ranking model on the current Agent data.
Do not train on the 19-task preference-ranking set.
Do not use the current Scorer as a sole training label.
Use ESCI as an external relevance research benchmark.
Use TREC as test-only independent benchmark.
Use Amazon Reviews as an external preference/rating benchmark, not a ranking training set.
```

## 9. Old Planner SFT adapter

The Planner SFT adapters are **not used** for ranking training:

- Planner SFT v0: rule-derived Planner Action adapter (185MB archive);
- Planner SFT v1: 53 real Qwen actions + rule labels, QLoRA (185MB archive);
- P14-C1: preference-operation QLoRA (173MB local archive).

They are retained as reproducibility artifacts because they document a negative result:
Action accuracy improved but Agent task success did not. They must not be repurposed as a
ranking model.

## 10. GPU handoff

No GPU training is handed off this round because the only defensible use of ESCI is an
external benchmark, not an Agent-catalogue reranker.

Prepared artifacts:

```text
Dataset A:
data/gold/esci_relevance_v1/{train,val,test}.jsonl.gz
data/gold/esci_relevance_v1/manifest.json
data/gold/esci_relevance_v1/benchmark_manifest.json

Dataset B:
data/gold/amazon_reviews_electronics_v1/{train,val,test}.jsonl.gz
data/gold/amazon_reviews_electronics_v1/manifest.json
data/gold/amazon_reviews_electronics_v1/benchmark_metrics.json

Independent TREC:
data/trec2025/2025-qrels.rec
data/trec2025/product-recommendation-2025-test-queries.tsv
data/trec2025/product-corpus.json.zst
data/gold/trec2025_product_recommendation_v1/manifest.json
```

If an external ESCI experiment is authorised later, the GPU runner must:

1. read the Dataset A train/val/test files unchanged;
2. use only train for fitting and val for early stopping;
3. keep the official test untouched;
4. report nDCG@10/nDCG_all/MRR@10 with both gain mappings and label the candidate-set scope;
5. never merge ESCI/TREC metrics with the 30-task Agent success metric.

The existing Qwen3-Reranker-4B and BGE model are on the AutoDL host; no model download is
needed. AutoDL is currently no-GPU mode, so no command was executed.

## 11. Agent E2E and P18

- The existing P14/P20 Agent evaluation remains the only Agent E2E protocol.
- Its 30-task labels remain **provisional**.
- ESCI relevance, TREC relationship labels, Amazon ratings, and Agent task success are
  **three separate evaluation classes** and must not be merged into one success rate.
- P18 Frozen Manifest untouched.
- No retrieval, Gate, Planner, or runtime code changed.

## 12. Verification

- Dataset A metadata: 0 missing product records;
- Dataset A split query groups disjoint;
- Dataset B target not in history; timestamps strictly ordered;
- Dataset B no unobserved item used as negative;
- TREC official labels unchanged; test-only;
- Inventory JSON validated as JSON;
- Local relevant tests pass (see final response).
