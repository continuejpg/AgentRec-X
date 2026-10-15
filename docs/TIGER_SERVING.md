# TIGER live serving engine (T-01)

Status: **verified end to end on real weights** (T-01C).  The accepted checkpoint was recovered
from the training host and hash-matched; `load_generator` succeeds on CPU and both retrieval
modes return real generative results through `TigerSourceTool`.

## Shape

```text
AgentRec-X (L2)                                backend venv (own process)
recommendation/backends/tiger_engine.py  ──►  backends/tiger_public/src/tiger_public/serve_worker.py
  TigerServingEngine                             Assets: load_generator(...) once
    .parent_asin_to_item_id(asin) -> int           CatalogueTrie + SidResolver
    .item_id_to_parent_asin(item_id) -> str        per request:
    .recommend(history_item_ids, k=20)               build_history_input -> TigerCaseScorer
        -> TigerServingResult                        -> retrieve(...)  [APPROXIMATE | CERTIFIED]
  identity (parent_asin) is re-attached HERE
```

Transport: one JSON object per line over the worker's stdin/stdout (`agentrecx.tiger.serve.v1`).
The worker is long-lived, so the generator loads once per process, not once per request.

## Boundary rules kept

* Only integer item ids and model outputs cross the boundary.  `parent_asin` is attached in
  `tiger_engine.py` from the trusted identity map and is never sent to the worker.
* The worker never sees a target, a metric, or any evaluation field.
* Nothing is invented: an unmappable item id, a non-contiguous rank, a missing `semantic_id`, a
  mode the backend did not actually run, a malformed frame, a dead worker or a timeout all
  raise.  Results are never shortened, substituted or relabelled.

## Constructing it (for T-02)

```python
from recommendation.backends.tiger_engine import (
    APPROXIMATE, CERTIFIED, TigerServingEngine, TigerWorkerConfig,
)

engine = TigerServingEngine(              # any identity map with the two directions
    identity,                             # e.g. SASRecInferenceEngine, or {asin: item_id}
    config=TigerWorkerConfig(device="cpu", k_max=20),
    mode=CERTIFIED,
)
with engine:
    report = engine.asset_report()        # device, load seconds, artifact hashes, resolver policy
    result = engine.recommend((11, 33), k=20)
```

`result` exposes `retrieval_mode`, `recommendations` (`item_id`, `parent_asin`, `rank`, `score`,
`semantic_id`), `checkpoint_sha256`, `sid_artifact_sha256`, `resolver`, so
`TigerSourceTool(engine)` works unchanged; `CandidatePlane(..., tiger_tool=TigerSourceTool(engine))`
registers it as `CandidateSource.TIGER`.

Default `required_frontier` is `k + |unique history|`, the frozen cohort's own rule
(`experiments/step26_gates.py` G16.1) - it is what lets a certified top-k survive a seen mask.

## Verified numbers (CPU, this machine)

| Fact | Measured |
| --- | --- |
| worker cold start (import + `load_generator`) | 442.9 s (second run 362.5 s; `load_seconds` 368.9 / 293.2) |
| `APPROXIMATE`, k=20, one frozen history | 1.54 s, `prefixes_expanded=1281`, 20 candidates, `status=APPROXIMATE` |
| `CERTIFIED`, one frozen history | 0.83-0.91 s, `prefixes_expanded=769`, `nodes_pruned_by_bound=1762`, `status=CERTIFIED`, `budget_exhausted=False` |
| scores | sequence log-probabilities, e.g. `-5.639 ... -7.02` |
| `TigerSourceTool` | real call returns triples; `score_kind=tiger_sid_sequence_logprob` |

Two behaviours T-02 must decide on (observed, not defects):

1. **CERTIFIED returns the frontier, not `k`.**  `retrieve` searches for `required_frontier`
   (`k + |unique history|`), so `recommend(k=20)` yields 50 rows for a 30-item history, and
   `TigerSourceTool.propose(limit=5)` returns 35 triples.  The head of the list is still the
   certified top-k; the tail is what survives a seen-item mask.
2. **The engine does not mask seen items** - by boundary rule the backend never masks.  Three of
   the 20 `APPROXIMATE(k=20)` candidates for frozen case 0 were items already in that user's
   history.  Whether TIGER's source layer masks is a T-02 decision; `history`/`two_tower` mask
   today while `catalog_search`/`similar_item` do not.

## Asset provenance (recovered)

`runs/tiger_public_2026/checkpoint/` now holds the accepted step-2.5 artifacts, taken from
`/root/autodl-tmp/gate_d_fp32_canonical_remediation/` on the training host and verified locally:

| File | sha256 | Against the H7 freeze |
| --- | --- | --- |
| `tiger.pt` (45,181,927 B) | `95e5cb6f816b…` | **match** |
| `generator_layout.json` | `eed4dffa…` | **match** (also byte-identical to a local re-derivation from `sid/`) |
| `tiger.json` (metadata) | `5d47bfd6…` | loader-enforced dependency check passes |
| `manifest.json` / `score_rule.json` | recorded alongside | provenance |

The other candidate checkpoints on that host (`gate_d_canonical_r1` = BF16, `…_smoke_out`,
`…_failed_run_…`) have the same size but **different** hashes and are *not* the H7 model.

## Historical asset blocker (resolved)

`load_generator` requires a checkpoint directory holding `tiger.pt`, `tiger.json` and
`generator_layout.json`.  Locally only the **step-2.4** side exists
(`runs/tiger_public_2026/{features,sid}`); the generator artifacts are absent, including
`runs/tiger_public_2026/checkpoint/generator_layout.json`, whose frozen hash
`eed4dffaccf8dfec…` is recorded in the H7 manifest.  Both entry points report this explicitly and
exit `3` rather than degrading.

Verified locally instead: the sid assignment (156,746 rows), the handoff catalogue, and their
sha256 values match the frozen H7 manifest; `CatalogueTrie` and `SidResolver` build from the real
`semantic_ids.json` + `layout.json`; the worker subprocess, the JSONL protocol and the fatal
startup frame all work over the real IPC.

## Tests

| File | Tier | Needs |
| --- | --- | --- |
| `tests/test_tiger_engine.py` | engine contract, fake transport | nothing (30 tests) |
| `tests/test_tiger_engine_real.py` | real artifacts, real worker process, real IPC | backend venv + sid/handoff |
| `tests/test_tiger_engine_real.py::test_real_recommendation_on_frozen_histories` | real generative inference (12 min: CPU start-up dominates) | restored checkpoint |

A skip is never a pass: the real-inference test skips with the missing path in its reason.
