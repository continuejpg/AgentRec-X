# Frozen benchmark protocol

This file fixes the first frozen evaluation set. It was created before any frozen-set
policy run; do not change tasks, expected assertions or policy settings in response to
frozen results. Any future change requires a new version/hash and a new protocol.

## Data hashes

`tasks/manifest.json` records the SHA-256 of the data files:

```text
dev.json    effeae24d760f6d8dfaa80b3137a4bb495e2092b00c4f2a64206b40168c2a932
frozen.json 17f21c9a641b518d6a1bbbb151dd5f87e90fe45720469175e2a7fd6d66fac28f
```

The runner verifies the manifest before running. Regenerating it is an explicit
`--manifest write` operation, not a normal run step.

## Fixed model and policy configuration

- Default candidate path: accepted SASRec + Two-Tower + BM25 lexical sources and
  full-precision RRF;
- k = 5; sources = `history,catalog_search`;
- BGE dense retrieval: **off**;
- ItemCF rerank: **off**;
- M7 grounded interactions: **on**, budget 2;
- R1 dynamic retrieval: **on**;
- M5 state integration: **shadow**;
- M8 trajectories: **on**;
- no Fine Ranker, no new recall source;
- accepted model/catalogue artifacts are loaded through the existing runtime and
  `ServiceSettings`; no temporary model path is hardcoded in benchmark code.

A/B/D require the existing structured LLM provider seam. If it is not configured, those
arms are reported as unavailable. No LLM is required for C.

## Frozen split

`tasks/frozen.json` contains 8 tasks and 3–4 turns each, covering the same scenario
families as the development split with different products and wording. It is not used to
change expected values, thresholds or prompt templates.

Run the frozen split only after the harness and the development split are frozen. Report:

- per-policy runnable/completed counts;
- deterministic assertions and failures;
- retrieval actions and final products;
- token usage where applicable;
- separate human relevance labels, if any.

## Metric separation

- next-item HR/NDCG/Candidate Recall are not shopping-task success metrics;
- deterministic task completion and product-role assertions are not user satisfaction;
- human relevance columns must be filled by an independent reviewer;
- missing human labels must remain missing, not inferred from Agent output.
