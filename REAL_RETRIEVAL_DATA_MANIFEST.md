# Real Retrieval Evaluation — Data Source & Filtering Manifest

**Status:** `INSUFFICIENT_GENUINE_REAL_SEARCH_QUERY_DATA`  
**Frozen proxy test:** `runs/p14d2_real_eval/formal_proxy_test.jsonl`  
**SHA-256:** `d21995647dbbf17c7b3fe1b617f8ad45d604b68d8fe4f50f3190b69be1e0ba53`  
**Frozen at:** `2026-11-10T03:50:06.604993+00:00` (before any test ranking)  
**Source data:** Amazon-C4 / MemRerank Electronics; MemRerank README declares **CC-BY-4.0**.  
**Important:** Amazon-C4 query text is a **review-derived proxy query**, not a logged real search query (P13-R report §2.2; C4 CSV has `query` + `ori_review`).

## 1. Source files and hashes

| Source | Role | Rows | SHA-256 |
|---|---|---:|---|
| `data/p14d1/raw/amazon_c4_user_query_dict_o3-mini_rewritten_v2.json` | Amazon-C4 original/rewritten query text (21,223 all-category entries; 1,546 Electronics) | 21,223 | see `runs/p14d2_real_eval/source_audit.json` |
| `data/p14d1/raw/train.jsonl` | MemRerank Electronics train, qid + positive product + history | 818 | see audit |
| `data/p14d1/raw/dev.jsonl` | MemRerank Electronics dev (the historical 135 source) | 135 | see audit |
| `data/p13r/electronics/tasks_raw.jsonl` | P13-R raw task reconstruction | 1,095 | see audit |
| `data/p13r/electronics/tasks.jsonl` | P13-R strict split, target timestamp/history | 1,069 (817/123/129) | see audit |
| `data/p13r/electronics/eval_tasks.jsonl` | P13-R train+dev evaluator view | 940 (817/123) | see audit |
| `data/p13/sports/amazon_c4_test.csv` | raw C4 schema evidence (`query`, `ori_review`); Sports, not used for evaluation | — | diagnostic only |
| `runs/p14d2/bge_index.npz` | fixed BGE index | 76,263 identities | `fce84987083452abd84b7fdbf09b1d55e35305e7c4983b03343034dca3f7a69d` |
| `models/bge-base-en-v1.5/pytorch_model.bin` | fixed BGE weights | — | `4b9d4a4b2d7ab06e3861861d5ee370974b31d17c0bb6af6f13d3dc79f33bbdcd` |

## 2. Funnel

### 2.1 Strict historical universe

```text
MemRerank Electronics raw rows (train 818 + dev 135 + test 143) = 1,096
→ P13-R raw task reconstruction                                  = 1,095
→ target parent_asin present in current 76,263 catalogue        = 1,087
→ strict split after cross-split user/qid/target removal         = 1,069
   (train 817 / dev 123 / test 129)
→ target present in current catalogue (strict)                  = 1,061
   (train 810 / dev 122 / test 129)
→ clean reserved proxy test                                     = 129
   (129/129 target coverage; never ranked in P13-R/P14-D1/P14-D2)
```

### 2.2 Query dictionary extension (not a clean split)

- 21,223 Amazon-C4 original/rewritten query texts over 31 categories.
- 1,546 Electronics entries; 1,535 targets in current 76,263 catalogue.
- Only 1,095 Electronics entries have a released MemRerank purchase-history/split; the other 451 have no released timestamp/split. They are **not** used in the formal test.
- 0 targets from other categories are in the current Electronics catalogue; no cross-category mixing.

## 3. Exclusion reasons

| Step | Removed | Reason |
|---|---:|---|
| MemRerank raw 1,096 → P13-R raw 1,095 | 1 | task reconstruction target/duplicate handling |
| raw 1,095 → target in current catalogue 1,087 | 8 | target identity not in the current 76,263 catalogue |
| raw 1,095 → strict 1,069 | 26 | dev/test user or target overlap with an earlier split (P13-R deterministic rule) |
| strict 1,069 → target in catalogue 1,061 | 8 | target identity not in current catalogue |
| query dict 1,546 Electronics → 1,535 catalogue targets | 11 | target identity outside current catalogue |
| query dict 1,535 → formal test 129 | 1,406 | only the P13-R strict **reserved test** was used as formal, never-ranked proxy test; train/dev are exposed diagnostics and were not called independent |

## 4. Why the old evaluation had 135

P14-D1 and P14-D2 used the **MemRerank Electronics Dev** split (135 raw / 123 strict). The same release also had Train (818 raw / 817 strict) and Reserved Test (143 raw / 129 strict). P14-D1 explicitly recorded that train was downloaded but unused; P13-R recorded that test was reserved and not run. Therefore 135 was a **split-selection choice**, not the number of available real queries.

Moreover, the old 135 are **C4 proxy queries** generated from target reviews with an observed purchase positive. They are not logged real search queries. Their historical BM25/BGE numbers are retained as historical proxy diagnostics and are not mixed into this report.

## 5. Label and time audit

- Label source: **observed positive purchase / Amazon-C4 positive item**.
- No click, add-to-cart, follow or judged-negative labels; all non-target products are **unjudged**.
- Strict tasks: 1,069/1,069 target timestamps; all histories strictly before target; target is not in its own history (`0` cases).
- Unique strict tasks: 1,069; unique users: 1,068; unique targets: 1,046; exact duplicate query texts: 0.
- True logged search sessions: **0** (no web session IDs; `sequence_id`/`qid` are evaluation-task identifiers, not sessions).

## 6. Formal-test scope

The 129-row formal file is a **frozen proxy-query benchmark**, not a genuine real-search benchmark:

- query is original C4 proxy text (not rewritten);
- target is the observed positive; target is not injected into retrieval;
- BM25 and BGE are unchanged, catalogue/BGE index/model unchanged;
- no test tuning; the file was frozen before evaluation.

All results must be labelled **Target Recall@K / Hit@K on C4 proxy queries**, not real search Hit@K.

## 7. External real-query data investigation (not used, no mixing)

| Candidate | Real queries? | Why not used here |
|---|---|---|
| JDsearch (`rucliujn/JDsearch`) | Yes; 171,728 test queries with candidates/labels | CC BY-NC-SA 4.0, anonymized term IDs, Chinese, 12.9M products; cannot directly feed the frozen natural-text BGE/current catalogue; separate external study required |
| ESCI / Shopping Queries Dataset | Yes; ~130k queries, ~2.6M manual relevance judgments | different Amazon catalogue/languages and separate licence; must be an independent external evaluation, not mixed with 76,263 |
| WANDS | Yes; ~480 queries | far below 5,000 and different Wayfair catalogue |
| CoShop adapter in repo | external benchmark | no licence/data in workspace; not used |

No external dataset was downloaded or mixed into the current catalogue. No genuine logged real-search Query–Target set is present locally.
