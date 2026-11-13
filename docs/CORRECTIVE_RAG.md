# Evidence-Aware Corrective RAG MVP

## 1. Existing Code Audit

Reused without a parallel state machine:

- `ElectronicsPlannerExecutor.execute` (RETRIEVE / RERANK / ASK / FINALIZE)
- `PlannerObservation` / `allowed_planner_actions` / `validate_action`
- `CandidateLedger`, `PreferenceAwareScorer`, `CandidateEligibilityEvaluator`
- `DecisionGate`, `GateBudget`, `no_verified_candidate_in_scope`
- Metadata sidecar and P15 Evidence Bundle / Claim Validator
- MCP `run_agent_turn` and existing Qwen `/plan` + `/explain`

Audit answers:

1. **Multi-round retrieval already exists** (initial search + one Gate retrieval round), but
   there was no evidence-driven query correction.
2. **No Query Rewrite / adaptive retrieval** existed in the repository.
3. The Gate exposes `NEED_MORE_CANDIDATES`, `NEED_MORE_PRODUCT_EVIDENCE`,
   `reason_code`, `unknown_fields` and budgets, but no explicit RETRIEVAL_GAP vs EVIDENCE_GAP
   classification.
4. The CandidateLedger de-duplicates source entries, but there was no explicit
   **stale-row reset across retrieval rounds**; this was a real bug: a second retrieval
   added candidates while FINALIZE could reuse scorer rows from the old pool.  Fixed in
   `_search` by clearing stale rows, eligibility and Gate state for the new candidate set.
5. The Planner can legally perform another RETRIEVE only while the search budget remains;
   `allowed_actions` now also stops advertising RETRIEVE when `retrievals_remaining=0`.
6. Needed new logic: a small deterministic **EvidenceCheck + query correction** module and
   one opt-in Executor recovery path.  No new Agent framework was added.

## 2. Open-Source Research & Reuse Decision

| Source | Actual URL | Licence / status | Decision |
|---|---|---|---|
| CRAG (official paper/code) | https://github.com/HuskyInSalt/CRAG · https://arxiv.org/abs/2401.15884 | GitHub API reports **no declared licence** (`license: null`) | **Pattern only**, no code copied. External web search + trained retrieval evaluator do not fit the fixed catalogue or no-new-model constraint. |
| Self-RAG (official) | https://github.com/AkariAsai/self-rag · https://arxiv.org/abs/2310.11511 | **MIT** (confirmed via GitHub API) | **Pattern only.** Requires a trained critic/generator with reflection tokens; incompatible with frozen Qwen3.5-9B and no training. |
| LlamaIndex RetryQuery / self-correcting query engines | https://docs.llamaindex.ai/en/v0.10.19/examples/evaluation/RetryQuery.html | LlamaIndex **MIT** | Not adopted: pulls in a large query-engine framework for one bounded retry. |
| Haystack QueryExpander | https://docs.haystack.deepset.ai/docs/2.30/queryexpander · https://github.com/deepset-ai/haystack | **Apache-2.0** | Not adopted: adds an LLM expansion component and framework dependency; deterministic hard-fact query is enough for the MVP. |
| LangGraph CRAG example | https://github.com/langchain-ai/langgraph (MIT); example community demo https://github.com/seanpoyner/CRAG_langgraph_DEMO | LangGraph MIT; example not used | Not adopted: existing Planner/Executor already owns the bounded loop. |

**Decision:** reuse the existing AgentRec-X loop and implement only a deterministic
`classify_evidence_gap` + `build_corrective_query` + one Executor-owned search.  No new
Python dependency, no vector DB, no new model, no web search, no copied code.

## 3. Architecture & Minimal Design

```text
RETRIEVE
  -> existing ledger/scorer/eligibility
  -> FINALIZE asks DecisionGate
       -> ALLOW_FINALIZE: unchanged
       -> NEED_USER_CLARIFICATION: unchanged ASK
       -> retryable evidence/search gap and policy=corrective:
             EvidenceCheck (RETRIEVAL_GAP / EVIDENCE_GAP / AMBIGUITY)
             -> one corrective query from missing hard facts
             -> existing search backend + CandidateLedger merge
             -> re-score / re-inspect
             -> return normal Observation
             -> next Qwen FINALIZE
             -> existing Gate re-validates
       -> exhausted/unrecoverable: bounded abstention
```

Properties:

- The corrective action is **Executor recovery**, not a hidden Qwen PlannerAction.
- The Planner still chooses whether to FINALIZE / RERANK / ASK afterwards.
- No UNKNOWN becomes verified; no Gate bypass.
- At most one corrective attempt (`corrective_max_attempts=1`).
- A search query already present in `search_queries` is rejected.
- Unverifiable facts (`price`, `stock`, `warranty`, etc.) never trigger query rewriting.
- Existing candidate ledger is preserved; new IDs are merged, not a replacement.
- Metadata/evidence provenance remains from the existing P15 bundle.

## 4. Implementation

| File | Change |
|---|---|
| `recommendation/control/corrective_rag.py` | New deterministic classification + bounded query builder |
| `experiments/p14c0/electronics_executor.py` | Opt-in corrective recovery; clears stale rows on new search; records corrective trace; preserves ledger/Evidence/Gate |
| `recommendation/demo/planner_contract.py` | Stop advertising RETRIEVE once search budget is exhausted |
| `recommendation/mcp/server.py` | `--corrective-rag` / `AGENTRECX_CORRECTIVE_RAG`; propagate to executor; return corrective trace |
| `experiments/p20_metadata/corrective_eval.py` | Real-catalog 27-task baseline/corrective evaluator |
| `experiments/p20_metadata/corrective_recovery_live.py` | Positive real-Qwen recovery with P15 final answer |
| `tests/test_corrective_rag.py` | Classifier, duplicate query, normal no-trigger, real-catalog positive recovery, UNKNOWN safety |
| `docs/CORRECTIVE_RAG.md` | This document |

Configuration:

```bash
# baseline (default)
python -m recommendation.mcp.server --metadata-mode enriched

# corrective
python -m recommendation.mcp.server --metadata-mode enriched --corrective-rag
# or: AGENTRECX_CORRECTIVE_RAG=1
```

No new dependency, model download or vector database.

## 5. Real Qwen Execution Trace

### Natural Query (the requested exact query)

```text
I need a wireless mechanical keyboard for my desk.
```

The new natural parser extracts (with source spans in the query):

```text
mechanical -> "mechanical"
wireless   -> "wireless"
```

MCP + Qwen corrective mode result: `RETRIEVE -> RERANK -> FINALIZE`, Gate `ALLOW_FINALIZE`, selected `B07WZFL74B`, corrective attempts **0**.  This is the correct no-gap behavior: the initial BGE retrieval already contained verified candidates, so Corrective RAG did not fire.  We did not manufacture a failure to force a recovery.

### Positive recovery (real Qwen Base, controlled task constraints)

```text
Qwen actions: RETRIEVE -> RERANK -> FINALIZE
Executor: one corrective retrieval after Gate NEED_MORE_PRODUCT_EVIDENCE
Qwen action: FINALIZE
Gate: ALLOW_FINALIZE
Selected: B07WZFL74B
Final answer: generated by the real Qwen /explain path
```

Evidence:

- initial query: `keyboard for desk` → no verified mechanical+wireless candidate;
- corrective query: `keyboard mechanical wireless`;
- new candidates: 17, including B07WZFL74B;
- new hard-verified candidates: 7;
- selected RK ROYAL KLUDGE RK61 wireless mechanical keyboard;
- final answer cites catalogue evidence for wireless/mechanical facts.

This is a **controlled task**: the query is natural but the mechanical+wireless hard
constraints are explicitly task-scoped and supplied to the executor; they are not hidden
evaluator labels.  The initial Qwen retrieval was broad and genuinely missed the target.

### Reasonable ask / abstain

- `p14e2_lite_0056`: `awaiting_clarification`, no corrective search (Gate ambiguity).
- `p14e2_lite_0057`: `awaiting_clarification`, no corrective search.
- `p14e2_lite_0060`: `no_verified_candidate_in_scope`; one corrective search was attempted
  after the evidence gap but did not create a verified candidate.  This is the required
  **extra search with no improvement** example, not a fake success.
- `controlled_unverifiable_price_warranty`: no corrective search; result is a bounded
  abstain because price/warranty are not in the fixed catalogue.

## 6. Before / After Evaluation

Fixed: Qwen3.5-9B `Qwen3.5-9B-ms` checkpoint, enriched sidecar, BGE index, 27 tasks
(20 P14-E2 dev + 5 P14-B3 Eval100 samples + 2 controlled tasks), same budgets and
generation settings.

| Metric | Baseline | Corrective |
|---|---:|---:|
| Task success | 25/27 | **26/27** |
| Recommend success | 16/18 | **17/18** |
| Correct ASK / abstain | 9/9 | 9/9 |
| Constraint violations | 0 | 0 |
| Corrective attempts | 0 | 2 |
| Corrective recovered | 0 | **1** |
| Corrective no improvement | 0 | 1 |
| Corrective trigger precision | 0% | **50%** |
| Actual search calls | 28 | 30 |
| Planner calls / Tool calls | 92 / 92 | 88 / 88 |
| Planner input tokens | 140,553 | 132,696 |
| E2E wall time | 337.2 s | 320.2 s |

Interpretation:

- Task success +1, entirely from the real recovery of the controlled keyboard task.
- The 2 corrective attempts: one genuine recovery (`B07WZFL74B`; 17 new candidates, 7 new hard-verified); one no-improvement (`p14b3_eval100_0093`; 10 new candidates, 0 new hard-verified, correct abstain).
- The aggregate Planner/tool call and token numbers are lower in this run, but **we do not interpret that as a stable efficiency improvement**; the actual corrective cost is +2 search calls and the 2 attempts above.
- Normal recommendations and ASK/abstain did not regress.
- CPU Rule: 23/27 → 24/27, same one recovery, 2 attempts, trigger precision 50%.

The previous 8-task table (7/8 → 8/8) is superseded by this 27-task run; more tasks exposed one fixed recommendation miss (`p14e2_lite_0051`) unrelated to Corrective RAG, and one insufficient-evidence label (`p14b3_eval100_0109`) that is correctly mapped to bounded abstention by the evaluator.

## 7. Regression & Tests

```text
tests/test_corrective_rag.py
tests/test_planner_reliability.py
tests/test_p14c0_contract.py
...
```

Result: **23 passed** on the focused corrective/planner suite; all affected suites are
listed in the final report.

P18: **1 failed, 5 passed**, the pre-existing freeze-manifest hash drift in
`recommendation/control/catalog_search.py`; not modified.

## 8. Limitations

- The controlled positive case supplies hard constraints as task-scoped inputs; it is not
  a fully natural benchmark label.  The exact natural query was tested through MCP and
  correctly did not trigger corrective retrieval because the initial candidates already
  had verified evidence.
- **No natural-language task in the 27-task set produced a genuine new recovery**; only
  the controlled keyboard task did.  This is a negative result for natural corrective
  value, not a success claim.
- One natural task did trigger corrective retrieval but produced no new hard-verified
  candidate (`p14b3_eval100_0093`); it is reported as no-improvement.
- The natural hard-constraint parser is conservative and pattern-based, not full NLU.  It
  supports explicit frames such as adjectives before the product noun, "with ...", "that
  is ...", and "should be ..."; it does not infer constraints from vague wording.
- The catalogue source-coverage check suppresses corrective search when no product of the
  requested type has all missing facts; this removed the invalid `0060` extra search.
- P14-E2/P14-B3 labels remain provisional and not human-verified.
- No P18 Reserved task was used; P18 hash drift remains a known pre-existing failure.

## 9. Adoption Decision

**Keep the MVP, not as a quality claim.** It is a small, opt-in, Gate-preserving recovery
mechanism with one real positive Qwen trace, one honest no-improvement trace, and no
constraint or ASK/abstain regression.

## 10. Next Steps

1. Add human-reviewed, fully natural positive tasks with explicit current-task constraints
   in the user query, and measure recovery over 20–30 tasks.
2. Replace the heuristic gap classifier with an independently validated evidence-sufficiency
   check; keep the existing Gate authoritative.
3. Decide whether a corrective attempt that adds no verified candidate should be suppressed
   earlier (e.g. cache failed query/evidence signatures) before expanding Corrective RAG.
