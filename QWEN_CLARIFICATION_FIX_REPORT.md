# Qwen Clarification Consistency Fix — Report

**Date:** 2026-11-10 (Asia/Shanghai)  
**Final verdict:** **GO** for the requested task-scoped clarification-consistency scope.

The real Qwen ASK path now treats an explicit current-purchase exclusion from the user's
clarification reply as authoritative for the resumed task, re-scores the preserved candidates
against that constraint, and prevents a final product with trustworthy evidence of the excluded
attribute. Historical M9 memory is not deleted; an explicit `Forget my wired preference.` still
uses the existing M9 retraction semantics.

**Important limitation:** the final browser-selected product in this run
`B00NFL0E2A` has **no trustworthy wireless evidence** in the catalogue. It was allowed only as an
existing Gate "limited recommendation" with `UNKNOWN` wireless evidence; the system does **not**
claim that this product is verified wireless. It has **no wired evidence**, so it does not violate
the current `avoid wired` exclusion. This limitation must travel with the GO result.

---

## 1. Root cause

The previous Qwen ASK → M9 → resume implementation had three separate gaps:

1. **No current-task constraint layer.**
   M9 correctly stored the reply as `wireless = prefer` and `wired = avoid`, but `avoid` is a new
   avoidance entry, not an explicit retraction. The old `wired = prefer` entry remained active.
   The resumed executor only used the active-M9 preference list; there was no separate representation
   of "for this purchase, wired is excluded" that takes priority over the old historical preference.

2. **The resume refreshed preferences but reused stale candidate-derived state.**
   The same executor and CandidateLedger were preserved, which is required, but
   `last_rows`, `_rows_by_rank` and `last_eligibility` had been produced before the clarification
   reply. On `FINALIZE`, the executor saw non-empty `last_rows` and skipped a fresh
   `_score_and_inspect`. Gate therefore received candidate rows whose preference conflicts/eligibility
   predated the reply. The previously selected `B091SF1545` was such a stale row.

3. **Product attributes were not carried into scorer rows.**
   `candidate_rows()` gave the scorer product attributes, but the scorer output did not retain the
   attribute evidence on each row. A task-level check could not see that
   `B091SF1545`'s title has direct `wired` evidence. Even with refreshed M9 preferences, the
   reconciled score row lacked the evidence needed to classify it as a task conflict.

The old result was therefore not evidence for changing M9/Gate wholesale; a task-scoped constraint
layer plus a true post-clarification re-evaluation was required.

---

## 2. Fix

### 2.1 Task-scoped current constraints

New `recommendation/preference/current_task_constraints.py`:

- Derives **current-task hard exclusions** from the current reply's M9 extraction:
  - `avoid wired headphones` → hard exclusion `attribute = wired`, provenance `"Please avoid wired headphones."`;
  - `prefer wireless headphones` → current soft preference;
  - explicit removals are not converted into current exclusions.
- `effective_preferences()` creates the current task's preference view:
  - removes the historical **positive** `wired` from the task view only;
  - adds the current `wired` negative;
  - never mutates or deletes the M9 entry.
- `CurrentTaskConstraints` is stored on the P17 clarification handle and copied into the Qwen
  continuation snapshot, so it survives the reply and is available on resume.
- The same reply extraction comes from the accepted M9 extractor through the P16 clarification
  skill (`reply_extraction`), not a separate hard-coded parser.

### 2.2 Product evidence matching

The executor now copies trusted `ElectronicsProduct.attributes` and `form_factor` into each scored
candidate row. `candidate_constraint_status()` then checks:

- explicit `wired` → task violation;
- explicit `wireless` → positive/soft match, **not** proof that an unknown `wired` value is absent;
- no connection evidence → `UNKNOWN`, never fabricated as satisfied;
- brand exclusions only when a trusted brand fact/evidence exists;
- negated evidence such as `not wired` is not treated as a positive wired match;
- product type/role continues through the existing Product Type/Role path (main vs accessory).

No single ASIN is special-cased. The matcher is attribute/value based over real catalogue evidence.

### 2.3 Post-clarification re-evaluation

`ElectronicsPlannerExecutor.refresh_for_task()`:

- retains the same executor, CandidateLedger, retrieval history and tool budgets;
- refreshes M9 historical preferences from a new `HistoryStateReader` read;
- creates the current-task effective preference list;
- re-scores the preserved candidate IDs with the latest preferences;
- populates task-constraint violations/unknown status;
- excludes task-violating rows from `last_rows`/planner/Finalize eligibility;
- returns an auditable `candidate_reevaluation` record with input count, excluded IDs and latency.

MCP `run_agent_turn` calls this method immediately before the resumed real Qwen `/plan`, so the
model cannot see stale candidate eligibility or stale Gate inputs.

### 2.4 Gate final consistency check

`DecisionGate` now receives task constraints and:

- filters candidate rows carrying `task_constraint_conflicts` before choosing a final candidate;
- returns a bounded/retry gate decision if every candidate is task-excluded;
- reports `task_constraint_status` (`none|verified|unknown|violated`) in its decision;
- the executor additionally checks the selected row's task conflicts immediately before
  presentation.

The existing hard/soft Gate rules and V0 scorer weights were not rewritten.

---

## 3. Before/after evidence

### Before the fix (recorded in `QWEN_ASK_RESUME_REPORT.md`)

- exact reply: M9 `added=2, removed=0`;
- old `wired — prefer` retained;
- stale candidate row `B091SF1545` (title `Earphones Wired Headphones Headphones Earbuds.`) was
  Gate-approved;
- final product was explicitly wired.

### After the fix

Real browser run `browser/summary.json`:

- M9 reply update: `added=2, removed=0`; old `wired — prefer` remains **active** in M9;
- current task hard exclusion:
  `wired`, source `"Please avoid wired headphones."`, source turn `qclar...`;
- candidate re-evaluation: `input=15, eligible=14, excluded=["B091SF1545"], violations=1`;
- no second clarification/handle; no `Forget` message required;
- final Gate: `ALLOW_FINALIZE`;
- final browser-selected product: `B00NFL0E2A`, title `Philmore Headphones.`;
  product-details evidence: **no wired evidence rows**; wireless evidence is `UNKNOWN`;
- final product is **not** an explicit-wired product.

Direct real-MCP run (`mcp_direct_cycle.json`):

- same task constraint and reevaluation;
- excluded `B091SF1545`;
- final selected product `B00M0KUF44`, title `RunPhones Wireless Headphones...`, catalogue
  attributes include explicit `wireless`;
- Gate `ALLOW_FINALIZE`, P15 evidence bundle present.

The two selected products differ because the browser and direct MCP use independent real search
paths/rankings; both satisfy the task exclusion. No rank was forced and no ASIN was preferred.

### Memory semantics comparison (`memory_semantics_compare.json`)

1. Current task exclusion:
   - M9 update `added=2, removed=0`;
   - old `wired` preference remains active in M9;
   - task constraint is stored separately.
2. Explicit `Forget my wired preference.`:
   - M9 update `removed=1, removal_directives=1`;
   - old `wired` becomes `inactive/removed`;
   - no task hard exclusion is synthesized from the Forget turn itself.

This is the required distinction: **avoid for this purchase ≠ permanent retraction**.

---

## 4. Real Qwen / Web E2E

Real Qwen3.5-9B Base on the existing RTX 4090, no adapter, real `/plan`, real P17 MCP,
real BFF/SSE/Web, isolated `profile_0056_ask`.

- Qwen emitted a real ASK:
  `I found conflicting preferences in your history (wired, wireless). Which one should take priority for this purchase?`
- Browser submitted the prescribed single reply:
  `For this purchase, I prefer wireless headphones for commuting. Please avoid wired headphones.`
- Qwen was called again with a new real `/plan`; it selected `FINALIZE`.
- Gate returned `ALLOW_FINALIZE`.
- Result: **one reply completed the task without the old extra "Forget" step**.

Screenshots:

- [first ASK](runs/web_demo/qwen_clarification_fix/browser/qwen-fix-first-ask.png)
- [after one reply / no follow-up](runs/web_demo/qwen_clarification_fix/browser/qwen-fix-after-reply.png)
- [final resumed Gate/product](runs/web_demo/qwen_clarification_fix/browser/qwen-fix-resume.png)

---

## 5. Tests and regression

| Test | Result |
|---|---|
| New task-constraint tests (`test_qwen_clarification_consistency.py`) | **6 passed** |
| Targeted P14-B1 Gate / P14-C0 executor / role / P15/P16/P17 / Qwen unit/BFF / Web unit | **93 passed**, 158.68 s |
| Full legacy Playwright E2E `01`–`07` | **9/9 passed**, 2.3 min |
| Updated real one-reply E2E `08` | **1/1 passed**, 1.2 min |
| Frontend `tsc --noEmit` + production build | **PASS** |
| CPU Rule / Qwen Replay separation | included in 9/9; no regression |
| Qwen Live real tests `07` | 2/2 passed in the 9/9 run |

The new tests cover:

- historical positive overridden only in the current effective task view;
- historical M9 entry not mutated;
- current exclusion vs M9 explicit retraction;
- wired/wireless/UNKNOWN/negated evidence;
- brand exclusion requiring trusted brand evidence;
- Gate filtering all task-conflicting candidates and selecting the non-conflicting candidate;
- no ASIN-specific logic in production code.

Performance:

- post-clarification candidate re-evaluation: **~1 ms** in the real direct run (`1.05 ms`),
  plus normal Qwen/BFF latency (browser total 69.9 s including real ASK + resumed Qwen + Gate).
- No additional retrieval was needed; CandidateLedger/retrieval budget was preserved.

---

## 6. Status matrix

| Required status | Result | Evidence |
|---|---|---|
| `TASK_CONSTRAINT_PRIORITY` | **PASS** | current hard exclusion `wired`; effective prefs override historical positive |
| `MEMORY_RETRACTION_SEMANTICS` | **PASS** | current avoid leaves history active; explicit Forget removed=1 |
| `POST_CLARIFICATION_REEVALUATION` | **PASS** | input 15 → eligible 14; B091 excluded; ledger/budget preserved |
| `PRODUCT_CONSTRAINT_CONSISTENCY` | **PASS** | final product has no trustworthy wired evidence; explicit wired product excluded |
| `GATE_FINAL_RECHECK` | **PASS** | Gate task filter + executor final check; unit all-invalid test blocks |
| `REAL_QWEN_RESUME` | **PASS** | real new `/plan`, `FINALIZE`; task constraints visible |
| `WEB_E2E` | **PASS** | 1/1 real browser; 9/9 legacy suite |
| `IDEMPOTENCY` | **PASS** | direct duplicate reply → `already_processed=true`, memory unchanged |
| `CROSS_USER_ISOLATION` | **PASS** | prior real BFF wrong-user handle → HTTP 409; handle-binding path unchanged by this fix |
| `LEGACY_REGRESSION` | **PASS** | 93 targeted tests; 9/9 Playwright; build |
| `FINAL_WIRELESS_EVIDENCE` | **UNKNOWN / LIMITED** | browser-selected B00NFL0E2A has no wireless/wired catalogue attribute; no claim of verified wireless |
| **Overall** | **GO** | no explicit current-exclusion violation in the final product |

**Non-blocking limitation:** `B00NFL0E2A` has `UNKNOWN` wireless evidence. The Gate's existing
soft-unknown policy allowed a limited recommendation; the UI/evidence must not describe it as
verified wireless. A future improvement could rank explicitly wireless-evidenced candidates first,
but that would be a separate ranking policy change, not required for this fix.

---

## 7. Artifacts

- Browser E2E summary: `runs/web_demo/qwen_clarification_fix/browser/summary.json`
- Browser evidence JSONL: `runs/web_demo/qwen_clarification_fix/browser/evidence_final.jsonl`
- Screenshots: `browser/qwen-fix-first-ask.png`, `qwen-fix-after-reply.png`, `qwen-fix-resume.png`
- Direct MCP transcript: `runs/web_demo/qwen_clarification_fix/mcp_direct_cycle.json`
- Task state/re-evaluation: `runs/web_demo/qwen_clarification_fix/current_task_state.json`
- Raw real `/plan`: `runs/web_demo/qwen_clarification_fix/plan_trace_direct.jsonl`
- M9 semantics comparison: `runs/web_demo/qwen_clarification_fix/memory_semantics_compare.json`
- Product evidence: `runs/web_demo/qwen_clarification_fix/selected_products_evidence.json`
- Targeted pytest log: `runs/web_demo/qwen_clarification_fix/pytest_targeted.log`
- Legacy regression: `runs/web_demo/qwen_clarification_fix/legacy_e2e.log`,
  `legacy_regression_summary.json`
- GPU/service: `runs/web_demo/qwen_clarification_fix/gpu_service_info.txt`
- Source diff: `QWEN_CLARIFICATION_FIX_SOURCE_DIFF.txt`

No commit/push/deploy was performed. Existing uncommitted work was preserved. No training/SFT,
V0 weight change, index rebuild, benchmark expansion, new Agent runtime, Reserved Test, ASIN special
case, or AutoDL instance lifecycle operation was used. The temporary model service/tunnel were
stopped after evidence collection; the existing GPU instance was left running.
