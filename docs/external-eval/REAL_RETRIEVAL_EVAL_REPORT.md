# AgentRec-X — Real Retrieval Evaluation Expansion

**Date:** 2026-11-10  
**Task:** audit and expand a genuine real Query–Target retrieval evaluation for the frozen BM25/BGE system.  
**Verdict:** **LIMITED / NOT SUPPORTED for a genuine real-search evaluation.** The workspace contains **0 genuine logged real-search Query–Target records**; it contains up to **1,535 C4 review-derived proxy query/positive-product pairs** in the current catalogue, of which **1,095** have released Amazon purchase histories and **129** form a clean never-ranked reserved test. The 5,000–10,000 target is **not met**, and cannot be met by cleaning these files without fabricating or relabelling data.

> This report separates **engineering completion** from **retrieval-quality evidence**. The frozen BM25/BGE proxy benchmark was executed correctly, but it is **not** a genuine real-search benchmark. No resume claim should present its numbers as real search retrieval quality.

---

## 1. Executive summary

1. **The old 135 was not a data ceiling.** It was the MemRerank/P13-R **Dev** split. The same release has Train 817 and Reserved Test 129 (strict), plus a 1,546-entry Electronics query dictionary; P14-D1 explicitly downloaded train but did not use it, and P13-R explicitly reserved test and did not run it.
2. **The old 135 are also not genuine search queries.** Amazon-C4 query text is a **review-derived proxy query** (P13-R §2.2; the C4 CSV has both `query` and `ori_review`). The target is an observed purchase, not a logged search impression/click. Under this task's own prohibition on review/history masquerading as real search, these records cannot be counted as genuine real Query–Target evaluation data.
3. **Maximum available proxy records:** 1,535 query/positive-product pairs in the query dictionary have targets in the current 76,263 catalogue; only 1,095 have released histories/splits; strict valid universe is 1,069 (1,061 targets present); clean reserved proxy test is 129.
4. **Maximum genuine logged real-search records:** **0** locally.
5. A frozen 129-row reserved **proxy** test was evaluated with unchanged BM25/BGE/index/model. BGE is much stronger on this proxy benchmark: Target Recall@100 50.39% vs 18.60% BM25; paired difference +31.8 pp, 95% CI [23.3, 40.3], McNemar p ≈ 1e-10. This is **proxy evidence only**.
6. Category stability cannot be established: all targets are Electronics, product-type metadata is missing for 100/129 test targets, and typed subgroups are tiny. No graded NDCG is reported because non-target products are unjudged.
7. This dataset is not suitable for a real-search resume claim. A genuine external evaluation needs JDsearch, ESCI/Amazon Shopping Queries, WANDS or equivalent real-search data with its own catalogue; it must not be mixed with the current 76,263 catalogue.

---

## 2. Data-source audit and filtering funnel

### 2.1 Source inventory

| Source | Rows/entries | Role |
|---|---:|---|
| `data/p14d1/raw/amazon_c4_user_query_dict_o3-mini_rewritten_v2.json` | 21,223 total; **1,546 Electronics** | Amazon-C4 original/rewritten query text; **review-derived proxy queries** |
| `data/p14d1/raw/train.jsonl` | 818 | MemRerank Electronics train (qid, positive product, purchase history) |
| `data/p14d1/raw/dev.jsonl` | 135 | MemRerank Electronics dev; source of the historical 135-query report |
| `data/p13r/electronics/tasks_raw.jsonl` | 1,095 | P13-R raw task reconstruction (train 817 / dev 135 / test 143 raw) |
| `data/p13r/electronics/tasks.jsonl` | 1,069 | P13-R strict split (train 817 / dev 123 / test 129) |
| `data/p13r/electronics/eval_tasks.jsonl` | 940 | strict train+dev evaluator view |
| `data/p13/sports/amazon_c4_test.csv` | diagnostic | C4 schema evidence: `query` is accompanied by `ori_review`; Sports, not used |

The MemRerank local README declares **CC-BY-4.0** and identifies Amazon-C4 / Amazon Reviews 2023 as source datasets. The formal test/source hashes and fixed index/model hashes are recorded in `runs/p14d2_real_eval/source_audit.json` and `frozen_proxy_test_manifest.json`.

### 2.2 Funnel

```text
MemRerank Electronics raw rows: 1,096 = train 818 + dev 135 + test 143
P13-R raw reconstruction:        1,095
target in current 76,263 cat.:   1,087 / 1,095 (99.27%)
strict split (no cross-split):    1,069 = 817 / 123 / 129
target in current catalogue:      1,061 / 1,069 (99.25%)
strict reserved proxy test:       129 (129/129 target coverage)
```

Query dictionary extension:

```text
21,223 C4 original/rewrite entries (31 categories)
→ 1,546 Electronics entries
→ 1,535 Electronics targets exist in current 76,263 catalogue
→ only 1,095 Electronics entries have released MemRerank history/split
```

The extra 451 Electronics query-dictionary entries have no released target timestamp/history/split and are **not** used in the formal test. No other-category targets are in the current Electronics catalogue, so there is no cross-category mixing.

### 2.3 Required audit questions

| Question | Actual finding |
|---|---|
| Raw genuine real Query count | **0 logged real search queries** locally. 21,223 Amazon-C4 proxy query texts (all categories); 1,546 Electronics. |
| Queries with a real associated product label | 1,095 Electronics proxy-query/observed-purchase pairs with released history; 1,535 query-dictionary Electronics pairs have targets in the current catalogue. |
| Target ASIN in frozen catalogue | 1,087/1,095 raw; 1,061/1,069 strict; 129/129 formal test. |
| Deduplicated independent queries | 1,069 strict tasks; 1,069 unique normalized queries; 0 exact duplicate query texts. |
| Independent users / sessions / products | 1,068 users; 1,046 unique targets; **0 genuine web sessions**; `sequence_id`/`qid` are task IDs, not sessions. |
| Category coverage | All targets are Electronics; query dictionary has 31 categories but only Electronics targets are in the current catalogue. Product-type metadata is null for 588/817 train, 94/123 dev, 100/129 test. |
| Timestamp completeness | 1,095/1,096 raw; 1,069/1,069 strict; 129/129 formal test. History strictly before target; target is not in its own history (0 cases). |
| Label behaviour | **Observed positive purchase** / Amazon-C4 positive item. No click/cart/follow labels and no judged negatives; all non-targets are unjudged. |
| Why only 135 before | P14-D1/D2 selected the Dev split (135 raw / 123 strict). Train 818 was downloaded but unused; reserved test 143/129 was never run. 135 was a split-selection choice, not the available maximum. |
| Can the source expand to 5,000–10,000? | **No.** Maximum proxy universe is 1,535 query-dictionary pairs / 1,069 strict history tasks / 129 clean reserved test; there is no genuine logged real query pool. |

### 2.4 Query authenticity finding (critical)

Amazon-C4 query text is **generated from the target review**, not a real search log. Evidence in this workspace:

- `P13R_REPORT.md` §2.2: “C4 Query 由 target review 生成，标识为 **proxy query**；原始搜索 Query … 不可得”.
- `data/p13/sports/amazon_c4_test.csv` has columns `qid, query, item_id, user_id, ori_rating, ori_review`; its `query` is a review-style sentence and the accompanying `ori_review` is the source review.
- Examples in `data/p14d1/raw/amazon_c4_user_query_dict...` are first-person review narratives (e.g. “I need filters … It's surprising how much dust they can collect …”).

Therefore the historical 135 are **proxy-query + observed-purchase** records, not real Query–Target search records. The project's earlier reports themselves labelled this `LIMITED`/proxy. This is the central reason the requested 5k–10k genuine real evaluation cannot be delivered from the current data without violating the task's explicit data-quality rules.

---

## 3. Frozen proxy diagnostic

### 3.1 Protocol

- **Formal test:** P13-R strict reserved test, **129 rows**, frozen before any ranking at `2026-11-10T03:50:06Z`.
- Test SHA-256: `d21995647dbbf17c7b3fe1b617f8ad45d604b68d8fe4f50f3190b69be1e0ba53`.
- Query: **original Amazon-C4 proxy text**, not rewritten.
- Target: only evaluator-side; not injected into either retrieval source.
- BM25 and BGE: frozen existing implementations; same 76,263 catalogue; same BGE index and weights; no training, no re-indexing, no parameter tuning.
- Top-100 retrieved for each method; metric is **Target Recall@K / Hit@K** for the single observed positive.
- **No graded NDCG**: other products have no relevance judgments.
- Exposed Train 817 (810 target-in-catalogue) and Dev 123 (122 target-in-catalogue) are reported separately as **non-independent diagnostics**, never merged into the formal test.
- Old 135 Dev results are historical and are not rerun/mixed.

### 3.2 Fixed configuration

| Component | Frozen value |
|---|---|
| Catalogue | 76,263 Electronics records; `data/p13r/electronics/catalog_meta.jsonl` |
| BM25 | `CatalogSearchSource(mode='bm25')`, unchanged |
| BGE | `CatalogSearchSource(mode='bge')` with `LocalBGEDenseIndex`; `BAAI/bge-base-en-v1.5` |
| BGE index | `runs/p14d2/bge_index.npz`; SHA-256 `fce84987083452abd84b7fdbf09b1d55e35305e7c4983b03343034dca3f7a69d` |
| BGE weights | SHA-256 `4b9d4a4b2d7ab06e3861861d5ee370974b31d17c0bb6af6f13d3dc79f33bbdcd` |
| Test tuning | none |
| Target injection | none |

### 3.3 Formal proxy test results (n = 129)

Rates are percentages; brackets are 95% user-grouped bootstrap CIs. This is **Target Recall@K on proxy queries**, not genuine real-search Hit@K.

| K | BM25 hits / n | BM25 Target Recall | 95% CI | BGE hits / n | BGE Target Recall | 95% CI |
|---:|---:|---:|---|---:|---:|---|
| 10 | 5/129 | **3.88%** | [0.78, 7.75] | 25/129 | **19.38%** | [12.40, 26.36] |
| 20 | 10/129 | **7.75%** | [3.10, 12.40] | 30/129 | **23.26%** | [16.28, 30.23] |
| 100 | 24/129 | **18.60%** | [12.40, 25.58] | 65/129 | **50.39%** | [41.86, 58.91] |

**Paired BM25 vs BGE (same 129 proxy queries):**

| K | BGE − BM25 | Paired bootstrap 95% CI | BGE-only / BM25-only | Intersection | Both missed | McNemar exact p |
|---:|---:|---|---:|---:|---:|---:|
| 10 | **+20 hits / +15.50 pp** | [7.75, 23.26] pp | 24 / 4 | 1 | 100 | 0.00018 |
| 20 | **+20 hits / +15.50 pp** | [8.53, 23.26] pp | 24 / 4 | 6 | 95 | 0.00018 |
| 100 | **+41 hits / +31.78 pp** | [23.26, 40.31] pp | 43 / 2 | 22 | 62 | ≈1e-10 |

BGE is significantly better than BM25 **on this proxy benchmark**. This does not establish a genuine real-search superiority.

### 3.4 Exposed diagnostics (not independent)

| Split | n | Method | Recall@10 | Recall@20 | Recall@100 |
|---|---:|---|---:|---:|---:|
| Train (exposed) | 810 | BM25 | 6.05% | 7.53% | 14.44% |
| Train (exposed) | 810 | BGE | 18.52% | 24.32% | 45.31% |
| Dev (exposed) | 122 | BM25 | 4.10% | 8.20% | 14.75% |
| Dev (exposed) | 122 | BGE | 13.93% | 20.49% | 50.00% |

These directions are consistent with the formal proxy test, but both splits are exposed and cannot be called independent. They are not mixed with the formal test.

### 3.5 Target coverage

- Formal test: **129/129** targets in the frozen catalogue.
- Raw P13-R proxy tasks: **1,087/1,095** (99.27%).
- Strict P13-R tasks: **1,061/1,069** (99.25%).
- Query-dictionary Electronics entries: **1,535/1,546** targets in the current catalogue.

### 3.6 Category and query splits (formal proxy test)

**Product type** (catalogue metadata; not a semantic category judgement):

| Product type | n | BM25 R@10/20/100 | BGE R@10/20/100 |
|---|---:|---:|---:|
| UNKNOWN / missing | 100 | 5/10/19 | 21/24/50 |
| headphones | 17 | 0/0/1 | 1/2/6 |
| speaker | 6 | 0/0/1 | 2/3/5 |
| keyboard | 3 | 0/0/1 | 0/0/2 |
| mouse | 2 | 0/0/1 | 1/1/1 |
| earbuds | 1 | 0/0/1 | 0/0/1 |

The product-type metadata is missing for **100/129** test targets. The typed groups contain at most 17 queries. **No stable category-level conclusion can be drawn.**

**Query length** (words):

| Bucket | n | BM25 R@10/20/100 | BGE R@10/20/100 |
|---|---:|---:|---:|
| <25 | 10 | 0/0/0 | 1/1/2 |
| 25–49 | 74 | 2/5/12 | 12/14/35 |
| ≥50 | 45 | 3/5/12 | 12/15/28 |

**Lexical query style** (fixed keyword heuristic, not a semantic truth):

| Style | n | BM25 R@10/20/100 | BGE R@10/20/100 |
|---|---:|---:|---:|
| feature-specific | 105 | 5/9/21 | 20/25/54 |
| generic | 24 | 0/1/3 | 5/5/11 |

All formal-test queries are `amazon_c4_proxy_from_target_review`; long (mean 51.7 words, range 12–189) and highly narrative, consistent with the review-derived provenance.

---

## 4. Failure analysis (formal proxy test only)

All 129 test rows were retained. At K=100, the target was found by both methods for 22 queries, by BGE only for 43, by BM25 only for 2, and by neither for 62. The full 107 non-both-success cases are in `failure_cases.jsonl`; no failed sample was deleted.

| Failure pattern | Count / evidence |
|---|---:|
| Target not in Top-100 for either method | **62** |
| BM25 miss, BGE hit (lexical shortfall) | **43** |
| BM25 hit, BGE miss (dense semantic shortfall) | **2** |
| Both hit at K=100 | **22** |
| Low literal query–title token overlap among failures | 32 |
| Target is BGE top-1 | 5 |

Representative evidence (query → target → observed top result):

1. **BGE succeeds, BM25 fails — semantic/product-type gap.** GoPro screw-mount query (`p13r...11081`: “screw mounts for my GoPro … well made … versatile”); target `[2 Pack] SEASKY … Thumb … Fixing Screws for Gopro`; BM25 misses, BGE rank 21; BGE top is `Sametop Tripod Mount Adapter Screw Mount …`. The query is long and review-like, but BGE recovers the accessory type.
2. **BGE rank 1, BM25 top is wrong accessory.** Mac mini enclosure/SSD query (`13659`); target `USB-C Hub with Hard Drive Enclosure for Mac Mini`; BGE rank 1; BM25 top is an iPod Touch case. This is a clean semantic success, not a target-injection artifact.
3. **BM25 succeeds on exact specs, BGE misses.** DDR4 query (`9863`: “8GB … 2933MHz … single RAM stick”); target `Micron … 8GB 3200MHz DDR4 … SODIMM`; BM25 rank 7, BGE misses Top-100. The query/target have a real spec mismatch (2933 vs 3200); this is a label/query-ambiguity and model/spec matching case, not simply “BM25 better”.
4. **BM25-only, target not unique.** Portable neck-fan query (`5013`); target `Elingmor Portable Neck Fan`; BM25 rank 12; BGE top is a handheld mini fan. Both products are plausible; one observed purchase does not make the other irrelevant.
5. **Both fail — review-derived ambiguous query.** “I want to buy two of these products because I bought one a year or two ago …” (`10423`) → target `iArkPower 4K60Hz HDMI Audio Extractor`; neither method retrieves it in Top-100. The query has no product-type anchor; the target is not uniquely inferable.
6. **Both fail — behaviour-label ambiguity.** A defective/replacement earbud narrative (`11890`) is paired with an earbud target; neither method retrieves the target in Top-100, and top results are unrelated products. A purchase-review sentence is not a clean positive search intent.
7. **Both fail — accessory/core product-type confusion.** Kindle Oasis durability query (`11994`) → target is a Kindle Oasis, but top results are Kindle cases/covers; both miss the core device.
8. **Both fail — target may not be the unique reasonable answer.** Webcam query (`12009`) → target is a 4K webcam, but both methods return other webcams/ring lights; the single observed positive is not the only plausible product.
9. **Both fail — brand/model and product-type gap.** Dual-screen laptop query (`1259`) → target ASUS ZenBook Pro Duo; top results are older laptops and an HDMI adapter; both miss.
10. **Both fail — accessory subtype ambiguity.** iPad holder query (`12304`) → target is a tablet tripod mount adapter; top results are iPad cases; both miss.

The failure taxonomy is therefore a mix of **proxy-query ambiguity / single-observed-positive labels**, **product-type/accessory mismatch**, **spec/brand mismatch**, **metadata gaps**, and **candidate-recall shortfall**. A miss is not proof that the target is irrelevant: all non-target products are unjudged.

---

## 5. Leakage, dedup and temporal audit

- **Split leakage:** strict Train/Dev/Test have **0** overlapping users, qids, targets, or normalized queries.
- **Test duplication:** 129 test rows = 129 unique users, targets, qids and normalized queries; 0 exact duplicate query texts.
- **Target leakage:** 0 test/validation targets appear in their own purchase history.
- **Temporal order:** all strict histories are strictly before the target timestamp; 1,069/1,069 strict tasks have a valid target timestamp.
- **Session isolation:** no genuine session IDs exist. The data provide task/sequence IDs, not web search sessions; no claim of session-level independence is made.
- **Temporal isolation status:** **LIMITED**. Target timestamps and histories are internally consistent, but the C4 query is not a logged search event; no vendor query timestamp exists. The test was never ranked before this run (P13-R reserved test), but it is not a time-isolated real search log.

---

## 6. External real-query data investigation

No local genuine real-search Query–Product set exists. The following public real-query candidates were identified but **not used and not mixed** with the 76,263 catalogue:

| Candidate | Real queries | Scale / labels | Licence / limitation | Decision |
|---|---|---|---|---|
| [JDsearch](https://github.com/rucliujn/JDsearch) ([paper](https://ar5iv.labs.arxiv.org/html/2305.14810)) | Yes, real JD.com queries | 171,728 test queries, candidate lists, click/cart/purchase labels | CC BY-NC-SA 4.0; anonymized term IDs, Chinese, ~12.9M products, non-commercial; cannot directly feed the frozen natural-text BGE/current catalogue | Separate external study required; not used here |
| [ESCI / Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) | Yes, real Amazon shopping queries | ~130k queries, ~2.6M manual relevance judgments | Different Amazon catalogue/languages; external licence/access terms; separate index required | Candidate for future external evaluation; not mixed |
| [WANDS](https://github.com/prabhant/WANDS) | Yes, Wayfair search queries | ~480 queries | Far below 5,000; different Wayfair catalogue | Too small for the 5k target |
| CoShop adapter in repo | external benchmark | — | No licence/data in workspace; adapter explicitly refuses absent data | Not used |

No external catalogue was downloaded or indexed. A future real-search evaluation must be a **separate external benchmark** with its own catalogue/index; it must not be merged with the current 76,263-catalogue proxy numbers.

---

## 7. Final answers to the six required questions

1. **Why were there only 135 before?**
   P14-D1/D2 used the MemRerank/P13-R **Dev** split (135 raw / 123 strict). The same release has Train 817 and Reserved Test 143 raw / 129 strict; P14-D1 recorded train as downloaded-but-unused and P13-R recorded test as reserved-but-not-run. 135 was a split-selection choice, not the dataset maximum. Moreover, those 135 are **C4 proxy queries**, not logged real search queries.

2. **What is the maximum number of reliable records the existing data can support?**
   - Genuine logged real-search Query–Target records: **0**.
   - C4 proxy query/observed-positive pairs with released history/split: **1,095 raw / 1,069 strict**; 1,061 targets in the current catalogue.
   - C4 proxy query/positive-product pairs in the query dictionary with current-catalogue targets: **1,535** (but 451 lack released history/split/timestamps).
   - Clean never-ranked reserved proxy test: **129** (129/129 target coverage).

3. **Did we reach 5,000 or 10,000?**
   **No.** The maximum proxy universe is 1,535 query-dictionary pairs, of which 1,095 have released histories and 129 clean reserved-test rows. There is no genuine real-search pool of 5,000–10,000, and no fabricated/cross-dataset mixing was used.

4. **Does BGE still beat BM25 on a larger independent sample?**
   On the clean independent **proxy** test (n=129), yes: BGE 19.38/23.26/50.39% vs BM25 3.88/7.75/18.60% at K=10/20/100; paired K=100 difference **+31.78 pp**, 95% CI [23.26, 40.31], McNemar p ≈ 1e-10. The exposed train/dev diagnostics show the same direction. But there is **no larger independent genuine real-search sample**; the real-search question remains unanswered.

5. **Are results stable across product categories?**
   **Cannot claim yes.** All targets are Electronics; 100/129 formal-test targets have no catalogue product-type metadata, and the typed groups have n ≤ 17. No cross-category comparison is possible. This is not a category-stability result.

6. **Is this metric set suitable for a resume?**
   **Not as a real-search retrieval result.** The proxy test is reproducible and properly labelled, but the query is review-derived and the test has 129 rows. It would be misleading to write “50.4% real-search Hit@100” or “BGE beats BM25 on real e-commerce search.” A resume-safe claim requires a genuine real-query benchmark (e.g. JDsearch/ESCI/WANDS with an independent catalogue), or a clearly labelled “proxy-query diagnostic” with explicit limitations. Current quality status is **NOT ESTABLISHED**.

---

## 8. Status matrix

| Item | Status |
|---|---|
| `SOURCE_DATA_AUDIT` | **PASS** |
| `GENUINE_REAL_SEARCH_QUERY_DATA` | **ABSENT (0)** |
| `PROXY_QUERY_TARGET_DATA` | **LIMITED (1,535 max; 1,095 with history; 129 clean test)** |
| `SAMPLE_SIZE_5K_10K` | **NOT MET** |
| `FROZEN_FORMAL_PROXY_TEST` | **PASS** (129 rows; SHA frozen before ranking) |
| `FIXED_BM25_BGE_CONFIG` | **PASS** (unchanged catalogue/index/model) |
| `LEAKAGE_DEDUP_AUDIT` | **PASS for split/dedup; LIMITED for true temporal/query provenance** |
| `BM25_BGE_PROXY_EVAL` | **COMPLETED (proxy only)** |
| `REAL_QUERY_RETRIEVAL_EVAL` | **NOT SUPPORTED by available data** |
| `FAILURE_ANALYSIS` | **PASS** (107 cases saved; no deletion) |
| `RESUME_SUITABILITY_FOR_REAL_SEARCH` | **NO** |
| **Overall** | **LIMITED / DATA-BLOCKED** |

---

## 9. Artifacts, reproducibility and scope

### Artifacts

- `REAL_RETRIEVAL_EVAL_REPORT.md` (this report)
- `REAL_RETRIEVAL_DATA_MANIFEST.md`
- `runs/p14d2_real_eval/source_audit.json` — source inventory, counts, hashes, funnel
- `runs/p14d2_real_eval/frozen_proxy_test_manifest.json` — freeze timestamp, SHA, index/model hashes
- `runs/p14d2_real_eval/formal_proxy_test.jsonl` — 129 frozen rows (SHA `d219956…`)
- `runs/p14d2_real_eval/proxy_test_results.json` — metrics/CI/paired/splits
- `runs/p14d2_real_eval/proxy_test_per_task.jsonl` — all 129 ranks/outcomes
- `runs/p14d2_real_eval/failure_cases.jsonl` — all 107 non-both-success cases
- `runs/p14d2_real_eval/expanded_proxy_diagnostic_per_task.jsonl` — exposed train/dev diagnostics
- `runs/p14d2_real_eval/build_audit_and_test.py`, `evaluate_proxy.py` — reproducible audit/evaluation

### Reproduce

```bash
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python runs/p14d2_real_eval/build_audit_and_test.py

wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python runs/p14d2_real_eval/evaluate_proxy.py
```

`evaluate_proxy.py` verifies the frozen test SHA before running and fails if the frozen file changed. It does not tune BM25/BGE or use test labels to alter retrieval.

### Scope and safety

- No query generated, rewritten, imputed or synthesized for this evaluation.
- No review/purchase text was relabelled as a search query.
- No dataset mixing; no external catalogue mixed with 76,263.
- No BM25/BGE weight, index, catalogue, V0, Gate or Memory change.
- No training, fine-tuning, re-indexing, test tuning, old Reserved Test or Agent benchmark.
- No commit/push/deploy was performed. Deliverables are left untracked and uncommitted per instruction.
