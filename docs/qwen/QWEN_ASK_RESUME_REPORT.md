# Qwen ASK → M9 → Resume — Real GPU Acceptance Report

> **Update 2026-11-10 — superseded for the task-constraint scope:** the follow-up
> [`QWEN_CLARIFICATION_FIX_REPORT.md`](QWEN_CLARIFICATION_FIX_REPORT.md) fixes the two limitations
> recorded below. The prescribed single reply now creates a task-scoped hard exclusion, the
> historical wired preference remains in M9, the preserved candidates are re-scored, and the real
> resumed Qwen turn finalizes a product with no trustworthy wired evidence. The original report
> remains the historical record of the pre-fix LIMITED state.

**Date:** 2026-11-10 (Asia/Shanghai)  
**Final verdict:** **LIMITED** — the real Qwen ASK branch is now exercised, but the exact prescribed
single user reply does **not** clear the old `wired` preference under the existing M9 semantics;
one additional **real browser-authored retraction** was required. The final Gate allowed
finalization, but selected a product whose catalogue title/evidence is explicitly wired. No Memory,
Gate or V0 semantics were changed to make this pass.

**Scope:** real Qwen3.5-9B Base, real GPU, real P17 MCP execution, real M9 writes, real BFF/SSE/Web,
real browser replies. No Rule-generated ASK, no HTTP double, no Replay, no hard-coded PlannerAction,
no forced FINALIZE.

---

## 1. Environment and model identity

| Item | Verified value |
|---|---|
| GPU | NVIDIA RTX 4090, 24,564 MiB, driver **570.124.04** |
| Model | **Qwen3.5-9B Base**, `adapter=None` |
| Model path | `/root/autodl-tmp/models/Qwen3.5-9B-ms` |
| Torch / CUDA | `torch 2.11.0+cu128`; CUDA available **True**; CUDA 12.8 |
| Model service | `training/agent_sft/model_server.py`, port 8009; temporary SSH tunnel `18009` |
| Planner | real `POST /plan` → valid `PlannerAction` → existing `run_planner` / `ElectronicsPlannerExecutor` |
| Runtime owner | existing P17 MCP server; **no second Agent runtime** |
| Service state after evidence | model service and tunnel stopped; GPU back to **1 MiB** |
| AutoDL lifecycle | no create/stop/restart/release/replace; existing instance reused |

`runs/web_demo/qwen_ask_resume/gpu_service_info.txt` contains the raw capture.

### Planner prompt correction (one generic correction)

The only Planner-prompt change is a general Decision-Gate contract rule; it contains no
wired/wireless special case:

- Gate `NEED_USER_CLARIFICATION` / `needs_user_input=true` → choose `ASK`; do not finalize.
- After `USER_CLARIFICATION_REPLIED` with no unresolved conflicts → use the candidates and
  `FINALIZE`, rather than repeating the question.
- `ASK` is forbidden when `needs_user_input=false` and `gate_action` is not
  `NEED_USER_CLARIFICATION`; a surprising budget/typo alone is not an ASK signal.
- FINALIZE only with candidates and no Gate user-input requirement.

| Prompt | SHA-256 |
|---|---|
| Before | `c84e25e18d1d7370809b2906ae08e62cb87f49bf8184e7c2134b81f0fd4feecb` |
| After | `d1d2c7f6dc9d342e187b337b908a8ae481534351955afdf7b695d8c3b639896f` |

See `planner_prompt_sha.txt`. The same final prompt was deployed to the remote model service and
verified by remote SHA.

### Qwen service robustness fixes (not business semantics)

Two service-level issues surfaced during real-GPU regression; both were fixed without changing
Planner/Gate/Memory semantics:

1. **Generation serialization:** `ThreadingHTTPServer` let the BFF health probe and a real turn
   interleave tokenizer/model generation. A process-local `generate_lock` now serializes generation.
2. **Single missing JSON closing brace:** Qwen Base occasionally emitted a valid Planner action
   object without the final `}`. The parser now completes only that single missing closing brace,
   preserving every model-authored field; any other malformed JSON remains a hard failure. The raw
   model text is retained in the planner trace. This repair is recorded as a service workaround,
   not as a rewritten action.

These fixes restored the existing real Qwen Live E2E (2/2) and are included in the final 9/9
regression.

---

## 2. Isolated test user and conflict construction

- Added an **opt-in-only** profile `profile_0056_ask` (user key `p14e2_lite_0056`) behind
  `AGENTRECX_WEB_DEMO_ASK_TEST_PROFILE=1`.
- Normal demo profiles remain exactly `profile_0049` and `profile_0057`; the normal regression
  asserts two profiles and uses no test-profile memory.
- `p14e2_lite_0056` is an existing benchmark fixture user whose history is ingested through the
  **real accepted M9 `process_turn` path** (`ShoppingCompanionMemory.ingest_sessions`). Its active
  state contains both:
  - `wired` — prefer, source turn `s1t0`, memory id `0d3d33ea23044b31b304c2e0`;
  - `wireless for my keyboard` — prefer, source turn `s0t0`.
- `HistoryStateReader` independently read both as **active**, and `conflicts()` returned
  `wireless vs wired / mutually_exclusive_positive_preferences`.
- The existing demo profiles 0049/0057 were not used for this ASK-memory test and their persisted
  memory was not modified as part of the acceptance.

Frozen test input:

```text
I need headphones. Can you recommend something suitable for me?
```

No test-specific hard-coded action or hard-coded clarification question was added.

---

## 3. Real Qwen ASK — PASS

The real model made four real `/plan` calls:

| Step | PlannerAction | Latency | Input tokens | Output tokens |
|---|---|---:|---:|---:|
| 1 | `RETRIEVE {"query":"headphones","source_policy":"balanced"}` | 3,684.65 ms | 867 | 86 |
| 2 | `RERANK {}` | 6,128.88 ms | 1,102 | 156 |
| 3 | `FINALIZE {}` | 2,766.28 ms | 1,093 | 88 |
| 4 | `ASK {"question":"I found conflicting preferences in your history (wired, wireless). Which one should take priority for this purchase?"}` | 3,854.53 ms | 1,339 | 94 |

The ASK followed a real Gate decision:

```text
Gate action: NEED_USER_CLARIFICATION
reason_code: preference_conflict_unresolved
allow_finalize: false
needs_user_input: true
```

The question is not Rule-generated, not a BFF rewrite, not a Gate-generated question, and not a
hard-coded test string. Raw request/response envelopes are in
`runs/web_demo/qwen_ask_resume/plan_trace_direct.jsonl` (lines for `needs_user_input=true`), and the
distilled transcript is in `mcp_transcript.json`.

**REAL_QWEN_ASK = PASS.**

---

## 4. Browser clarification UI — PASS

Real browser E2E `e2e/tests/08-qwen-ask-resume.spec.ts` selected the isolated profile, entered
Qwen Base Live, submitted the frozen query, and displayed the real ASK card:
`I found conflicting preferences in your history (wired, wireless)...`

Evidence:

- `runs/web_demo/qwen_ask_resume/browser_final/qwen-ask-first.png`
- `browser_final/evidence_final.jsonl` check `qwen_ask.real_ask`
- `browser_final/summary.json` (1 passed, 2.8 min)
- `sse_ask_raw.txt` (29 real SSE events including `clarification`, `done`,
  `stop_reason=awaiting_clarification`)

**CLARIFICATION_UI = PASS.**

### No-reply safety — PASS

After ASK and before any reply:

- no `memory-update` block;
- Gate remains `NEED_USER_CLARIFICATION`;
- M9 `active`/`inactive`/`history_status` unchanged (the BFF JSON `fetched_at` timestamp was the
  only difference in the first capture; the comparison excludes it);
- direct MCP resume attempt fails with
  `clarification_reply_required: no real user reply was persisted yet`;
- no FINALIZE is emitted.

**NO_REPLY_SAFETY = PASS.**

---

## 5. First real browser reply and the M9 semantic limit

The acceptance text was submitted through the real browser input and real HTTP POST:

```text
For this purchase, I prefer wireless headphones for commuting. Please avoid wired headphones.
```

BFF confirmation source: `web_ui_explicit_user_reply`. M9 result:

```text
added=2  superseded=0  removed=0  skipped_duplicates=0  removal_directives=0
```

The two added entries are:

- `wireless headphones for commuting` — **prefer**;
- `wired headphones` — **avoid**.

However the old `wired` entry remains **active**:

```text
wired — prefer — active — memory_id=0d3d33ea23044b31b304c2e0 — source_turn_id=s1t0
```

This is an existing M9 semantic behavior: `avoid` is a new avoidance preference, **not** an explicit
retraction. `process_turn` reported `removed=0` and `removal_directives=0`; `conflicts()` therefore
still returned `wireless vs wired`, and the Gate still required user clarification. The real Qwen
resume therefore produced a **second real ASK**, not a FINALIZE. This was not forced or hidden.

> **Failure against the literal one-reply acceptance criterion:** after the exact prescribed first
> reply, the old wired preference is still active. Changing Memory semantics to make an `avoid`
> entry retract an older `prefer` entry was explicitly out of scope and was not done.

**M9_REAL_UPDATE = PARTIAL** for the exact prescribed single reply (real write succeeded; the old
wired preference was not removed). **MEMORY_STATE_VERIFIED = NOT ACHIEVED** at this point.

This behavior is also visible in the browser:
`browser_final/qwen-ask-after-first-reply.png` and the `qwen_ask.followup_ask` check.

---

## 6. Second real-browser reply → M9 update → verified state

To complete the task truthfully, the same isolated test user submitted a second **real browser
message** (not a fabricated reply, not a direct DB edit):

```text
Forget my wired preference.
```

That is an explicit M9 retraction and the accepted `process_turn` path applied it:

```text
added=0  superseded=0  removed=1  skipped_duplicates=0  removal_directives=1
```

Final M9 state (independent `HistoryStateReader` / BFF `/api/preferences`):

| State | Value | Polarity | Status |
|---|---|---|---|
| Active | `wireless headphones for commuting` | prefer | active |
| Active | `wireless for my keyboard` | prefer | active |
| Active | `from a keyboard` | prefer | active |
| Active | `wired headphones` | **avoid** | active |
| Inactive | `wired` | prefer | **removed** |

- old wired positive preference is no longer active;
- the audit/inactive entry remains visible as removed;
- user key is `p14e2_lite_0056` throughout;
- no cross-user state is included.

**MEMORY_STATE_VERIFIED = PASS after the second real user reply; PARTIAL/FAIL for the exact
prescribed single-reply-only flow.**

---

## 7. Qwen Resume — PASS (technical continuation)

The continuation is implemented in the existing MCP server: a clarification handle retains the
**same executor instance**, CandidateLedger, candidate IDs, tool observations, retrieval/Gate
budgets, clarification ID and original query. On reply, only the M9-derived preference list and
memory history are refreshed, then a **new real Qwen `/plan` call** runs. It never reuses an old
PlannerAction and never starts the turn from a new unrelated query.

After the first reply, the real resume PlannerAction was:

```text
ASK
```

After the second (retraction) reply, the real resume PlannerAction was:

```text
FINALIZE
```

The final Gate decision:

```text
action: ALLOW_FINALIZE
allow_finalize: true
reason_code: evidence_sufficient
selected_candidate_id: B091SF1545
unknown_fields: ["wireless"]
candidate_presentable_ids: [B091SF1545, B00M0KUF44, B00NFL0E2A, ...]
```

P15 evidence/`evidence_bundle` was present. The browser displayed a product card and Gate card.

**QWEN_RESUME = PASS** (same task/ledger/budget, real new inference).
**GATE_FINALIZATION = PASS** (existing Gate returned ALLOW_FINALIZE; no Gate change).

### Final recommendation preference compliance — LIMITED / FAIL

The Gate-selected product is **B091SF1545**, whose real catalogue title is:

```text
Earphones Wired Headphones Headphones Earbuds.
```

Its own P13R evidence includes the `wired` attribute at a lexical span. The final Gate reason is a
*limited* recommendation with `unknown_fields=["wireless"]`; it allowed finalization even though the
user's current active state contains `wired headphones = avoid` and `wireless = prefer`. No attempt
was made to rewrite the selected item or bypass the Gate. This is a real final-recommendation
quality/compliance limitation, not a fabricated successful recommendation.

Evidence:

- `selected_product_gate.json` (catalogue title, wired evidence, UNKNOWN fields);
- `browser_final/qwen-ask-resume.png` (UI shows the selected wired-title product);
- `browser_final/summary.json` check `qwen_ask.gate_final` (Gate action `ALLOW_FINALIZE`, 11 cards).

**PREFERENCE_CONSISTENT_FINAL_RECOMMENDATION = LIMITED/FAIL.**

---

## 8. Negative and isolation checks

| Check | Result | Evidence |
|---|---|---|
| No reply → no M9 write / no finalize | **PASS** | direct MCP `clarification_reply_required`; BFF preference state unchanged; no `memory-update`; Gate `NEED_USER_CLARIFICATION` |
| Duplicate reply (same `turn_id`) | **PASS** | direct MCP `already_processed=true`; M9 memory unchanged; no duplicate event |
| Invalid/expired clarification handle | **PASS** | direct MCP `invalid_or_expired_clarification_id`; BFF HTTP 409 |
| Cross-user handle | **PASS** | BFF session for `profile_0057` with a `profile_0056_ask` handle → HTTP 409 `clarification_not_pending` |
| Old wired preference after retraction | **PASS after second reply** | independent M9 reader; old wired `removed`, absent from active preferences |
| Cross-user Memory contamination | **PASS** | all direct/browser runs bound to `p14e2_lite_0056`; other profiles untouched |

Raw negative capture: `negative_api.json`, `negative_summary.json`, `sse_ask_raw.txt`.

---

## 9. Regression

| Suite | Result |
|---|---|
| Full Playwright E2E `01`–`07` | **9/9 passed**, 5.2 min |
| New real ASK browser E2E `08` | **1/1 passed**, 2.8 min |
| Targeted pytest (`P15/P16/P17/Qwen Live/BFF/planner/web demo`) | **67 passed**, 148.29 s |
| Frontend `tsc --noEmit` + production build | **PASS** |
| Qwen Live real `07` after service fixes | **2/2 passed** |
| CPU Rule Live / Qwen Replay | included in the 9/9 pass; no mode regression |
| P17 default tool surface | still 7 tools when Qwen is not configured; 8 with `run_agent_turn` only when Qwen is configured |

Note: an earlier Qwen Live run exposed the model-side malformed-JSON / concurrent-generation issues
described in §1. Those were fixed and the final clean run passed. Earlier retry logs are retained
for audit, but the final result is the clean 9/9 run in `regression_e2e_final.log`.

---

## 10. Acceptance matrix

| Required status | Result | Note |
|---|---|---|
| `REAL_QWEN_ASK` | **PASS** | real model raw output `ASK`; relevant conflict question |
| `CLARIFICATION_UI` | **PASS** | real browser SSE/UI; screenshot |
| `M9_REAL_UPDATE` | **PARTIAL** | exact reply added 2; old wired not retracted |
| `MEMORY_STATE_VERIFIED` | **PASS after second real reply / PARTIAL for one-reply flow** | old wired removed; wireless active |
| `QWEN_RESUME` | **PASS** | same executor/ledger/budget/clarification; new `/plan` |
| `GATE_FINALIZATION` | **PASS** | existing Gate `ALLOW_FINALIZE`; no Gate edit |
| `NO_REPLY_SAFETY` | **PASS** | no write, no finalization |
| `IDEMPOTENCY` | **PASS** | duplicate `already_processed=true`; no duplicate memory |
| `CROSS_USER_ISOLATION` | **PASS** | wrong-user handle HTTP 409 |
| `WEB_E2E` | **PASS** | 9/9 regression + 1/1 ASK E2E |
| `PREFERENCE_CONSISTENT_FINAL_RECOMMENDATION` | **LIMITED / FAIL** | Gate-selected wired title despite `wireless`/`avoid wired` |

**Overall: LIMITED.** The core real-model ASK → M9 → continuation branch exists and was exercised,
but the prescribed one-reply acceptance cannot be satisfied by the current M9 retraction semantics,
and the existing Gate's limited-finalize path selected a preference-conflicting wired product. No
fake, Rule substitute, Replay, forced action, Gate/Memory semantic change, or preferred-result
selection was used.

---

## 11. Files and evidence map

Code/test deliverables:

- `recommendation/mcp/server.py` — Qwen ASK continuation handle (same executor/ledger/budget),
  real prefs refresh, generic conflict signal, MCP `clarification_id` resume argument;
- `web_demo/config.py` — opt-in isolated `profile_0056_ask`;
- `web_demo/demo_flow.py` — second clarification returned to the browser, no silent finalization;
- `web/lib/types.ts`, `web/lib/chat.ts`, `web/components/DemoApp.tsx`,
  `web/components/ClarificationCard.tsx` — follow-up clarification UI;
- `training/agent_sft/planner_prompt.py` — one generic Gate/ASK contract correction;
- `training/agent_sft/model_server.py` — generate lock + single-missing-brace parser robustness;
- `recommendation/demo/planner_http_proposer.py` — opt-in request/response trace env;
- `e2e/tests/08-qwen-ask-resume.spec.ts` — real GPU browser ASK→M9→resume E2E;
- `e2e/global-setup.ts` — allow explicitly configured extra isolated profile for this suite;
- `scripts/run_web_demo.ps1` — forward the test-profile flag.

Evidence:

- `runs/web_demo/qwen_ask_resume/browser_final/` — final 1/1 browser evidence, screenshots,
  payloads, summary;
- `browser/` — earlier real browser/API captures including raw SSE and negative checks;
- `plan_trace_direct.jsonl` — raw real `/plan` request/response envelope for every direct call;
- `direct_cycle.json` / `mcp_transcript.json` — direct real MCP ASK, two real replies, resume, Gate;
- `m9_events_and_states.json`, `negative_summary.json`, `selected_product_gate.json`;
- `regression_e2e_final.log`, `regression_evidence_final.jsonl`, `regression_pytest.log`;
- `gpu_service_info.txt`, `plan_trace_0049.jsonl`, `cleanup_*.txt`;
- `QWEN_ASK_RESUME_SOURCE_DIFF.txt`.

No commit, push, deploy, model training/SFT, adapter load, V0/Gate/Memory semantic change, Reserved
Test, or AutoDL lifecycle operation was performed. The temporary model service and tunnel were
stopped after evidence collection; the existing GPU instance was left running.
