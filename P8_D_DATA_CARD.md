# P8-D Data Card: Multi-user Source-Policy Development Dataset

## Identity

- Dataset version: `agentrecx.p8d.multiuser_policy_dataset.v1`
- Path: `runs/p8_d/multiuser/`
- Built: 2026-11-08 (Asia/Shanghai)
- Purpose: development analysis of history/catalog source complementarity and
  Source Policy ranking effects; **not a training set**.

## Files

| File | Meaning |
|---|---|
| `cases.jsonl` | 300 case definitions: user, train history, validation target, source, canonical/extended flag |
| `candidate_cache.jsonl` | Real history and Hybrid BM25+BGE Top-50 candidate snapshots, query, source budgets, cache provenance |
| `policy_results.jsonl` | Per-case policy replay results, source reachability, features, metrics, strict outcomes |
| `summary.json` | Aggregate results and learnability status |
| `state*.sqlite3` | Local runtime state from the experiment (no user data beyond execution) |
| `../baseline_manifest.json` | Frozen model/index/policy/source-budget baseline metadata |

## Population

- 157 canonical P6 Dev cases (after excluding 3 target-in-history cases).
- 143 additional eligible Dev users from the same temporal protocol and the same
  `sha256(user_id) % 5` user split.
- Total evaluated: 300; skipped: 0.
- Frozen Holdout users selected: **0**.
- Target in history: **0**.
- Test target (`items[-1]`) is not written into the dataset and is not used for queries.
- Target position: validation target (`items[-2]`) only.

## Input construction

- History = `parent_asins[:-2]`, chronologically ordered by the preprocessing artifact.
- Query = last two history items' categories/title; target never used.
- `query_source` is stored per case.
- History source: real Serving V1 history tool, Top-50.
- Catalog source: real Hybrid BM25+BGE, Top-50; catalogue/index from P8-B.
- The same immutable candidate snapshot is used for all three policies.
- Policy replay uses the exact `weighted_reciprocal_rank_fusion` function used by the Agent.

## Labels and metrics

- Primary label: `validation_parent_asin` (next-item relevance proxy).
- This is **not** a human search-satisfaction label and not a search-query relevance label.
- Metrics: Recall@10, NDCG@10, MRR@10 over the union of the two Top-50 candidate pools.
- Reachability is reported separately; a policy cannot recover a target absent from both pools.
- Strict winner = unique best NDCG@10 greater than zero on a reachable sample.
- Oracle labels use validation targets and are analysis-only, never SFT labels.

## Source Policy replay

| Policy | history weight | catalog weight |
|---|---:|---:|
| balanced | 1.0 | 1.0 |
| intent_focus | 1.0 | 2.0 |
| history_focus | 2.0 | 1.0 |

Formula: `score(d) = Σ_s weight[s] / (60 + rank_s(d))`.

## Baseline controls

- Serving V1: unchanged; history source includes its existing SASRec/Two-Tower/BM25 fusion.
- BGE: `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
- Index: `runs/p8_bge/bge_base_dense.npz`; 156,746 × 768.
- Catalogue: current 156,746-product Amazon Sports catalogue.
- M5 / Eligibility / Role Gate: unchanged.
- No retraining, no Holdout, no Gold/Evaluator changes.

## Known limits

- Only 15/300 targets are reachable in the two Top-50 pools.
- Catalog source contributes only one reachable target; history dominates.
- Query is a history-derived proxy, not a real user search query.
- Current shopping-intent evidence is separate, synthetic, and has no human
  Query-Product relevance labels; see `runs/p8_d/current_intent/retrieval_eval.json`.
- The dataset is suitable for development/diagnosis, not for claiming independent
  generalization or for automatically minting SFT-v1 positive labels.
