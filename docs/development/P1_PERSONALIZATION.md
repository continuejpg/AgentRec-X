# P1 — Opt-in Memory Retrieval and Behavior Rerank

> **Status: implemented and verified on the current Web CPU path.**
> Both features are independent server-side flags, default **OFF**.  No V2 code,
> SASRec/Two-Tower/RRF, M10A/M10B, frontend or README was modified.

## 1. Feature flags

| Flag | Env var | Default | Effect |
|---|---|---|---|
| Memory-aware retrieval | `AGENTRECX_MEMORY_AWARE_RETRIEVAL` | `0` | Adds bounded M9-derived auxiliary queries to the **formal** CPU retrieval, fuses by rank, then uses the normal scorer/Gate. |
| Behavior-aware rerank | `AGENTRECX_BEHAVIOR_AWARE_RERANK` | `0` | Sends a bounded short-term behavior snapshot into the formal CPU scorer. |

The BFF parses the env vars into `web_demo.config.Settings`; `MCPManager` passes explicit
CLI flags to the stdio child, and the child also accepts the env vars for direct MCP use.
`/api/health` and `/api/demo/config` expose the two booleans for local diagnosis.  A request
body cannot turn either flag on.

`M9 → V0 PreferenceAwareScorer` is unchanged and still runs when both flags are OFF.

## 2. Memory-aware retrieval

```text
recommend_products(query, session_id)
  -> HistoryStateReader / get_active_preferences
  -> prepare_preferences (canonical positive/negative)
  -> effective_preferences (task hard exclusions override M9 positives)
  -> at most 2 auxiliary queries: "<effective_query> <positive value>"
  -> base CatalogSearchSource.search(limit=20)
     + auxiliary CatalogSearchSource.search(limit=10) per query
  -> reciprocal_rank_fusion([base, aux...]) (existing implementation)
  -> normal V0 Scorer -> Eligibility -> Task Constraints -> Budget -> Gate
```

Properties:

- The base query is always retrieved first and all its candidates remain in the fused
  ranking; memory can add candidates but cannot replace/evict the base set.
- Only ACTIVE positive M9 values become auxiliary queries.  Task-scoped hard exclusions are
  applied through `effective_preferences`; negative/withdrawn values are never query terms.
- At most 2 auxiliary queries, 10 hits each, 40 merged candidates.
- BM25, BGE and hybrid all work because the executor uses the existing `CatalogSearchSource`.
- Auxiliary retrieval failure is recorded (`fallback=true`, `error`) and the base candidates
  are used; the turn is not failed.
- The metadata records: enabled, queries, base/auxiliary counts, `new_item_ids`, merged
  count, extra latency and error.
- No Category-scope filter from M3-C is used.

Measured on the real catalogue (BM25 `studio headphones` + M9 `USB`):

```text
base=20, aux=10, merged=22, new_item_ids=2, extra_latency=44.03 ms
```

On the real BGE path (same query/preference):

```text
search_mode=bge, base=20, aux=10, merged=23, new_item_ids=3, extra_latency=105.32 ms
```

## 3. Behavior-aware rerank

```text
browser click / Like / Dislike
  -> existing web_demo.behavior BehaviorStore event
  -> short-term profile (window + half-life + event weights)
  -> bounded BehaviorSnapshot v1
       schema, session_id, product_interests (id/score/count/last_event),
       attribute_interests (name/score/count), no raw titles/event text
  -> BFF recommend_products(behavior_snapshot=...)
  -> PreferenceAwareScorer:
       behavior_score = 0.25 * product_signal + 0.10 * attribute_signal
       capped at [-0.30, +0.30]
       final_score = existing V0 score + behavior_score
  -> Task Constraints / Eligibility / Budget / Gate unchanged
```

- Explicit Like has a larger profile score than a Click (`2.0` vs `1.0` in the existing
  behavior profile); Dislike is negative.  The snapshot is versioned and bounded to 5
  products / 5 attributes.
- A behavior contribution is a **soft** term.  It never bypasses hard constraints,
  `hard_unknown`, eligibility, budget, or the Decision Gate.
- Behavior is never written to M9.  The snapshot is session-scoped and MCP rejects a snapshot
  whose `session_id` differs from the request session.
- No behavior events / empty profile => zero contribution and the old ordering.
- Replay uses the committed durable result and never re-reads behavior.  Cold Resume
  recomputes the current server flag and current session profile; an already committed reply
  is never re-ranked.
- The existing `/api/behavior/shadow` remains a separate preview and is not the formal path.

Real BM25 evidence (`wireless studio headphones`, Like on a presented card):

```text
active_product_signals=1, active_attribute_signals=1
candidates_with_behavior_signal=20
top_behavior_score=0.233333
top_behavior_candidate_id=<presented candidate>
```

The formal `recommend_products` path returned `behavior_rerank` metadata; behavior was not
only present in the shadow endpoint.  The unit matrix additionally proves an OFF→ON order
change in the real executor: baseline selects `B0BASE1`, behavior selects `B0BASE2`.

## 4. Four flag combinations

| Memory | Behavior | Verified behavior |
|---|---|---|
| OFF | OFF | Base retrieval; no `memory_retrieval`/`behavior_rerank` metadata; original M9 scorer/Gate still active. |
| ON | OFF | Auxiliary retrieval adds candidates; behavior contribution absent. |
| OFF | ON | No auxiliary retrieval; behavior contribution changes the formal scorer. |
| ON | ON | Both metadata blocks present; memory adds candidates and behavior contributes independently. |

The `test_personalization_features.py` executor matrix covers all four; real E2E covers
OFF/OFF, ON/ON, and OFF/ON; the real memory E2E also runs with both flags on.

## 5. Files

- `web_demo/config.py`: flags + env parsing.
- `web_demo/mcp_manager.py`: explicit child CLI flags.
- `web_demo/behavior.py`: `BEHAVIOR_SNAPSHOT_SCHEMA`, `build_behavior_snapshot`.
- `web_demo/app.py`: server-owned snapshot creation, health/config feature fields.
- `web_demo/demo_flow.py`: behavior snapshot in initial chat and reply paths; feature metadata on products.
- `web_demo/reply_executor.py`: behavior snapshot through claim/execute; feature metadata in durable result.
- `web_demo/reply_assembly.py`: expose feature metadata on reply responses.
- `recommendation/mcp/server.py`: flags, behavior snapshot validation, memory query expansion, metadata.
- `experiments/p14c0/electronics_executor.py`: auxiliary retrieval/RRF, behavior snapshot, audit metadata.
- `recommendation/preference/scorer.py`: bounded behavior contribution.
- `experiments/p14b1/diagnose.py`: pass behavior snapshot into the existing candidate-row builder.

## 6. Tests and compatibility

- `tests/test_personalization_features.py`: flags, query expansion, snapshot privacy,
  click-vs-like bounded scoring, four-combination executor matrix.
- `tests/test_web_personalization_e2e.py`: real FastAPI + MCP + BM25 memory retrieval,
  real behavior event -> formal rerank, both-off compatibility.
- Real BGE smoke: memory retrieval on `search_mode=bge` (not a BM25 substitute).
- Existing M4, P0 task constraints, warm/cold, replay, requirement-update and demo-price
  suites are re-run; see the final P1 report.

## 7. Remaining limits

- Behavior-aware rerank is wired to the CPU `recommend_products` path; the optional Qwen
  `run_agent_turn` path intentionally does not consume behavior snapshots yet.  Flags are
  OFF by default, so the Qwen path is unchanged.
- Memory retrieval is bounded to 2 auxiliary queries / 40 merged candidates and cannot
  recover an item that neither base nor auxiliary lexical/dense retrieval finds.
- A behavior snapshot is evaluated at reply time for a Cold Resume; only a committed replay
  is frozen.  The flags themselves are server-level, not request-level.
- No human quality labels or production A/B; these are engineering integration results, not
  a recommendation-quality claim.
