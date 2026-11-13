# Planner Reliability & Efficiency — Bounded RETRIEVE Fix

## Scope

This change addresses two reproducible Planner/Executor defects found in the existing
Electronics executor.  It does not add a framework, train a model, change the Planner
prompt, enlarge the action budget, or modify Metadata.

## Diagnosis

Ten P14-E2 dev tasks were run with the real Qwen3.5-9B planner on the AutoDL RTX 4090 and
the existing shared executor (BGE, Memory, CandidateLedger, Eligibility, Decision Gate,
Metadata enriched).  No P18 Reserved task was used.

Baseline: 9/10 task success; 1 `budget_exhausted`; 1 redundant RETRIEVE; 0 hard
constraint violations; 0 false finalize.

Two defects explained the failure:

1. **Retry budget OR-bug.**  After a failed FINALIZE the executor treated
   `NEED_MORE_PRODUCT_EVIDENCE` as retryable if *either* product-evidence rounds or
   retrieval rounds remained.  When evidence rounds were exhausted but one Gate
   retrieval remained, it asked for another search; that search hit the separate search
   budget and ended as `budget_exhausted` instead of a bounded no-verified outcome.
2. **Advertised actions disagreed with the action contract.**  `RETRIEVE` and `RERANK`
   were advertised as allowed after a completed RERANK, while `validate_action` rejects
   RETRIEVE then and repeating RERANK adds no evidence.  In the intermediate fixed build
   this produced a RERANK loop in the no-match task.

The observation also lacked a bounded candidate summary and a candidate-set delta after
RETRIEVE, so the Planner had only `candidate_count` to work with.

## Minimal Fix

- `recommendation/demo/planner_contract.py`
  - `allowed_planner_actions` now advertises:
    - ASK + RETRIEVE while not reranked;
    - ASK + FINALIZE + RERANK when candidates exist and not reranked;
    - ASK + FINALIZE after rerank (no RETRIEVE/RERANK).
- `experiments/p14c0/electronics_executor.py`
  - Per-action retry budget:
    - `NEED_MORE_PRODUCT_EVIDENCE` retries only while product-evidence rounds remain;
    - `NEED_MORE_CANDIDATES` retries only while Gate retrieval rounds remain;
    - exhausted budget terminates as `no_verified_candidate_in_scope`, not a fabricated
      recommendation and not `budget_exhausted` from a redundant search.
  - RETRIEVE now reports bounded `candidate_evidence`, `last_retrieve_query`,
    `retrieval_feedback` (new candidate count, candidate-set change, same-query flag,
    remaining search rounds), and the same counters in `m5_state`.
  - RERANK reports `candidate_evidence`, top-level `eligibility`, and preserves retrieval
    feedback.
  - Failed Gate feedback includes `gate_next_step` (`RERANK` for more product evidence,
    `RETRIEVE` for more candidates).
- `tests/test_planner_reliability.py`: regression tests for observation feedback, the
  advertised-action contract, and the evidence-budget termination.
- `experiments/p20_metadata/planner_diagnose.py`: the real-Qwen diagnostic runner.

No action budget was increased (`MAX_STEPS=8`, Gate budget and search budget are
unchanged).  No UNKNOWN attribute is treated as satisfied.  The Decision Gate still owns
finalize authority.

## Action Contract Check (`p14e2_lite_0060`)

- The second RERANK is a **PlannerAction**, not an internal Gate recovery.  The first
  failed FINALIZE returns a Gate observation (`NEED_MORE_PRODUCT_EVIDENCE`) through
  normal `replace(observation, reranked=False, gate_next_step="RERANK")`; the next
  Planner call therefore legitimately advertises RERANK.  The Gate never executes RERANK
  itself.  After the RERANK completes, the stale RERANK hint is consumed and replaced by
  `gate_next_step=FINALIZE`, so the next call cannot be told to RERANK while only
  FINALIZE is allowed.  This stale hint was a real contract bug found during follow-up
  audit and is covered by the test.
- `allowed_actions` is now checked against `validate_action` for every scenario:
  no candidates / candidates / clarification / reranked.  ASK is no longer advertised
  after candidates exist unless the Gate signals a clarification need; that was the
  remaining advertised-but-invalid action.
- `gate_next_step=RERANK` is only emitted together with candidates and a reset
  `reranked=False`, so RERANK is a legal next action.  It is a hint only; the Executor
  does not skip Gate checks.
- The displayed `eligibility.verified` value is **category-level eligibility only**,
  not hard-constraint verification.  It now carries `scope="category_eligibility_only"`
  plus `hard_constraint_verified/unknown/ineligible`.  In `0060`,
  `eligibility.verified=5` means five candidates passed the category projection while
  `hard_constraint_verified=0`; terminating with
  `no_verified_candidate_in_scope` is therefore not a semantic contradiction.  The Gate
  remains the hard-constraint authority.

## Before / After (10 real Qwen dev tasks)

| Metric | Before | After |
|---|---:|---:|
| Task success | 9/10 | **10/10** |
| Recommend success | 7/7 | 7/7 |
| Correct ASK / abstain | 2/3 | **3/3** |
| Budget exhausted | 1 | **0** |
| Redundant RETRIEVE | 1 | **0** |
| Hard constraint violations | 0 | 0 |
| False finalize | 0 | 0 |
| Model calls | 31 | 31 |
| Tool calls | 52 | 54 |
| Planner-only Wall Time | 136.4 s | 139.7 s |
| Planner input tokens | 33,042 | 45,710 |

The affected task `p14e2_lite_0060` changed from:

```text
RETRIEVE > FINALIZE > RETRIEVE > FINALIZE > RETRIEVE
status = budget_exhausted
```

to:

```text
RETRIEVE > RERANK > FINALIZE > RERANK > FINALIZE
status = no_verified_candidate_in_scope
```

The Gate still returned `NEED_MORE_PRODUCT_EVIDENCE`; the bounded executor stopped with
no selected product.  This is an abstention, not a recommendation success.

## Real Qwen Live

Natural query (no appended “you must finalize”):

> I need earbuds I can use while working out.

Result: `RETRIEVE -> RERANK -> FINALIZE`, `ALLOW_FINALIZE`, selected `B004K6R43G`,
12 product-evidence items, 4 valid explanation claims, 93.4 s end-to-end.

Latency decomposition:

- MCP startup/connect + BGE: 39.5 s;
- `run_agent_turn`: 45.5 s;
  - three Planner calls: 12.1 s;
  - `/explain`: 15.1 s (5,834 input tokens, 428 output tokens, below the 512 cap);
  - search/rerank/evidence/Gate and tool overhead: ~18.3 s;
- `get_product_info`: 3.6 ms.

The `/explain` cap is 512 new tokens on the AutoDL service; the 428-token response was not
truncated.

## Tests

```text
tests/test_planner_reliability.py
tests/test_p14c0_contract.py
tests/test_p14c0r_product_role.py
tests/test_p14b1_decision_gate.py
tests/test_p14c1.py
tests/test_p15_rag.py
tests/test_p16_skills.py
tests/test_p17_mcp.py
tests/test_qwen_live_mcp_integration.py
tests/test_electronics_metadata_enrichment.py
tests/test_docs.py
```

Result: **162 passed, 1 skipped**.

P18: **1 failed, 5 passed**.  The failure is the pre-existing freeze-manifest hash drift in
`recommendation/control/catalog_search.py`; no frozen file or Reserved Test was changed.

The 25-task reusable Dev checklist is in `docs/DEV_EVAL_CHECKLIST.md`.  Labels are
provisional: P14-E2 dev is `automated_checked`, P14-B3 Eval100 is
`pending_human_review` / `human_verified=false`.

## Remaining Issues

- `p14e2_lite_0056`/`0057` still attempt FINALIZE before the Gate returns
  `NEED_USER_CLARIFICATION`; the final outcome is correct but one step could be saved.
- `p14e2_lite_0060` now abstains correctly but still uses two RERANK calls.
- The P14-E2 dev set has no explicit wired-headphones task; `0056` covers the
  wired/wireless current-task conflict instead.
- Candidate evidence raises Planner input from 33,042 to 44,779 tokens across 10 tasks;
  this is an observability/quality tradeoff, not a forced finalize.
