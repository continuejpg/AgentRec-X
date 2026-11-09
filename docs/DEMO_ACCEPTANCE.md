# Demo acceptance and experiment modes

This is a short operator note, not a new evaluation framework. The default is the
accepted Serving V1 path; every experimental signal below is opt-in.

## Default demo

```bash
python -m uvicorn recommendation.api.app:app --host 127.0.0.1 --port 8766
```

Defaults: SASRec / Two-Tower / lexical catalogue sources, accepted full-precision RRF,
no BGE index, no ItemCF rerank, no Fine Ranker. The default profile is unchanged.

## Fixed demo profile

Use this single profile for Demo A (correction/withdrawal) and Demo B (uncertainty,
grounded examples, preference, rerank, withdrawal):

```text
AGENTRECX_M7_GROUNDED_ACTIONS=on
AGENTRECX_M7_BUDGET=2
AGENTRECX_R1_DYNAMIC_RETRIEVAL=on
AGENTRECX_STATE_MODE=shadow
AGENTRECX_STATE_DB=<demo-state.sqlite3>
AGENTRECX_M8_TRAJECTORY=on
```

No BGE or ItemCF artifact is required for Demo A/B; they run on the default
three-source retrieval path.

Optional experimental profiles use these switches. Paths are supplied by the operator,
never hardcoded in the project:

```text
AGENTRECX_CATALOG_SEARCH_MODE=bm25|bge|hybrid
AGENTRECX_DENSE_INDEX=<bge_base_dense.npz>
AGENTRECX_BGE_MODEL_PATH=<local BAAI/bge-base-en-v1.5 snapshot>

AGENTRECX_ITEMCF_RERANK=on
AGENTRECX_ITEMCF_ARTIFACT=<itemcf_csr.npz>
```

Enabled modes fail loudly if an artifact is missing or incompatible:
`dense_retrieval_unavailable` / `itemcf_rerank_unavailable`.

## Artifact requirements

- ItemCF CSR: format `agentrecx.itemcf_cosine.csr.v1`, `num_items=156746`,
  directed pairs `33,276,356`, full float64 similarity, and a mapping SHA matching the
  current `Sports_and_Outdoors_mappings.json`.
- BGE dense: format `agentrecx.dense_catalog.v3`, 156,746 × 768 BGE-base-en-v1.5
  vectors, model revision + catalogue/mapping fingerprints recorded in the artifact.
  The serving index must match the current catalogue identities and text fingerprint.
- The current artifacts were built from the real Sports catalogue and mappings; a
  different catalogue/mapping/model version must be rejected rather than mixed.

## Core reproducible cases

1. Demo A: `I need a complete water filter` -> complete filtration devices, not unrelated
   sleeping pads or lanterns; then `actually I want a replacement mouthpiece for a water
   filter` -> the replacement part only, not a bottle bite valve; then `forget the
   replacement mouthpiece requirement` -> restore the complete-filter scope, and after a
   service restart `show more options` must not resurrect the withdrawn part.
2. Demo B: `I don't know which type of water filter I need.` with M7/R1 on -> grounded
   catalogue examples and a weight comparison. User chooses `Lighter weight` -> M5
   explicit `weight minimize` event; final order changes from
   `[B08PRXXX7B, B08SLRGK9S]` to `[B08SLRGK9S, B08PRXXX7B]`; `forget the weight
   preference` restores the baseline order.
3. Uncertainty must not create a preference before the user answers. The initial turn
   records only task context and the grounded explanation event.

## Metric and identity rules

- canonical identity: `parent_asin` ↔ 1-based `item_id`, `PAD = 0`;
- evaluation filters seen items in the integer domain and retains a repeated target
  (`seen - {target}`);
- R5's old temporary helper compared ASIN strings to integer history and reported 11.55%
  Candidate Recall; the corrected clean pool is 11.30% (226/2000). Do not mix the two
  denominators;
- M9 used a stricter clean pool and did not beat A1; Fine Ranker remains offline only;
- next-item HR/NDCG is not shopping satisfaction and is not a human relevance label.

## P3 manual review

The 20 executed real Sports queries are listed in [P3_MANUAL_REVIEW.csv](P3_MANUAL_REVIEW.csv).
It contains 56 final candidate rows with query, ASIN, trusted title, category, trusted
attribute evidence, and empty human fields. Current state: **0 human labels, 56
unreviewed**. The CSV must be filled by a human reviewer; Agent output must not be
copied into the label columns.

## M9 decision

A0 = RRF; A1 = RRF + compact ItemCF; A2 = lightweight MLP pilot.

- A1 improved over A0 and was independently reproduced;
- A2 did not reliably beat A1 (HR@20 Δ −0.0005, NDCG@20 Δ +0.00172, CIs crossing zero);
- keep A1 ItemCF opt-in; do not wire A2 into Serving; no new recall source or model.

## Project status and limits

Validated engineering pieces: M5 unified state/retraction and replay, M6.1 belief-aware
ranking, M7 grounded ASK/COMPARE/EXPLAIN, M8 trajectory/choice events, R1 dynamic
retrieval, opt-in R2 BGE retrieval, and compact ItemCF in-pool rerank. CoShop-inspired
preference construction is limited to grounded examples, comparison/explanation, explicit
user confirmation, and R1 rerank/retrieval. It is not a CoShop simulator, and it does not
infer confirmed preferences from explanations or Agent guesses.

The project is ready for a scripted engineering interview demo, not production-quality
claims. Real human relevance labels, purchase/satisfaction labels, and a fresh independent
A2 evaluation are still incomplete.
