# AgentRec-X — Real-Agent Task Quality Baseline

## 1. Task Suite

30 natural-language tasks were selected from existing P14-E2 Dev and P14-B3 Eval-100 rows.
No P18 Reserved task was used.

- 20 P14-E2 Dev tasks;
- 10 P14-B3 Eval-100 tasks;
- 19 recommend;
- 6 clarification/ASK;
- 3 no-verified-match;
- 2 insufficient-evidence/abstain;
- 2 memory-conflict ASK tasks;
- multi-constraint/no-match tasks include over-ear + mechanical combinations.

All tasks use their existing public query, history, evaluator and memory fixture.
Labels are **provisional**:

- Dev: `p14e2_dev_automated_checked_provisional`;
- Eval-100: `p14b3_eval100_pending_human_review_provisional`.

No independent human review was performed. `automated_validated` from P15 is not semantic
truth and is not treated as a task gold label.

Artifacts:

- `runs/p20_agent_eval/task_suite.json`;
- `runs/p20_agent_eval/task_arm_baseline.json`;
- `runs/p20_agent_eval/task_arm_corrective.json`;
- `runs/p20_agent_eval/complete_shadow.json`.

## 2. Arms

| Arm | Metadata | Corrective RAG | Verifier | Harness | Tasks |
|---|---|---|---|---:|---:|
| A Baseline | enriched | off | off | existing direct Planner–Executor evaluation | 30 |
| B Corrective | enriched | on | off | same direct harness | 30 |
| C Complete Shadow | enriched | on | shadow | MCP `run_agent_turn` | 10 representative |

A/B were run with real Qwen3.5-9B Planner through the existing HTTP `/plan` service.
C used the existing MCP server with real Qwen `/plan` + `/explain` and Shadow Verifier.
C is a representative subset, not a full 30-task run.

The direct A/B harness and MCP C harness do not load memory identically. This is explicitly
reported rather than hidden: on the C subset, direct A/B scored 9/10 and C scored 9/10, but
different tasks failed.

## 3. Task-Level Results

| Metric | A Baseline | B Corrective | C Complete Shadow (10) |
|---|---:|---:|---:|
| Task success | 28/30 = 93.3% | 28/30 = 93.3% | 9/10 = 90.0% |
| Recommend success | 17/19 | 17/19 | 5/5 |
| ASK/no-match/abstain | 11/11 | 11/11 | 4/5 |
| Memory-conflict success | 2/2 | 2/2 | 1/1 |
| Hard-constraint violations | 0 | 0 | 0 |
| Corrective attempts | 0 | 2 | 0 |
| Corrective recovered | 0 | 0 | 0 |
| Corrective no improvement | 0 | 2 | 0 |

A/B use the same 30 tasks; C uses a 10-task representative subset, so C's success rate is
not directly comparable to A/B's 30-task rate.

### A/B task failures

- `p14e2_lite_0049` (expected recommend): both A and B ended
  `no_verified_candidate_in_scope`; no candidate was finalized. The Planner performed two
  RETRIEVE/RERANK cycles but the executor produced no presentable candidate.
- `p14e2_lite_0054` (expected recommend): both A and B finalized `B09TB1SCJP`, but the
  evaluator's correct-top-1 target was different. This is a ranking/selection miss, not a
  hard-constraint violation.

Corrective RAG triggered twice on the no-match tasks `0093` and `0098`, added two retrieval
attempts, and recovered nothing. On this natural task set it did not improve task success.

### C task failure

- `p14b3_eval100_0093` (expected no verified match): MCP `run_agent_turn` returned
  `is_error`, no structured content, with:
  `qwen_planner_error: planner_invalid_action: unknown planner action ''`.
  The task is counted as a failure. This is a real planner/tool error, not a Verifier error.

C 0049 succeeded through MCP even though direct A/B 0049 failed. That is a harness/memory
loading discrepancy, not evidence that the Verifier improved selection.

## 4. Natural Claim–Evidence Results (C)

C produced 17 natural claims on the five selected-product tasks:

| Metric | Count |
|---|---:|
| Claims total | 17 |
| P15 valid claims | 16 |
| P15 rejected claims | 1 |
| Qwen Verifier SUPPORTED | 14 |
| Qwen Verifier INSUFFICIENT_EVIDENCE | 2 |
| Verifier CONTRADICTED | 0 |
| Tasks whose valid final claims all supported | 3/5 selected-product tasks |

Important natural findings:

1. `p14e2_lite_0049 claim_3`
   - Claim: “This selection aligns with your preference for wireless earbuds.”
   - Cited: `mev_4d761...`, whose source is `"I also prefer wireless."`
   - P15: `automated_validated`.
   - Verifier: `INSUFFICIENT_EVIDENCE` (“cited preference does not match the claim”).
   - This is a natural preference-provenance/semantic mismatch.

2. `p14b3_eval100_0001 claim_3`
   - Claim: “The headband is made of polyester material.”
   - Cited: `pev_07`, whose span is only `"material"` (a field label, not the value).
   - P15: `automated_validated`.
   - Verifier: `INSUFFICIENT_EVIDENCE`.
   - This is a natural metadata-field overclaim.

3. `p14e2_lite_0042 claim_3`
   - Claim: “The brand of this product is unknown.”
   - P15: rejected (`UNKNOWN claim does not name a bundle-unknown field: brand`).
   - Verifier: `INSUFFICIENT_EVIDENCE`.
   - This demonstrates the existing P15 structural guard working.

Normal supported claims include wireless/over-ear/noise-cancelling assertions and IPX5
sweatproof assertions. They remain provisional until human review.

## 5. Memory / Gate / Corrective Key Traces

- Memory conflict ASK tasks `0056`/`0057`: A/B both `awaiting_clarification` and success;
  C `0056` also `awaiting_clarification` and success.
- `0056` C: Planner actions `RETRIEVE → RERANK → FINALIZE → ASK`; no product selected.
- Hard-constraint violations: 0 in A/B and C.
- `p14e2_lite_0060` C: `RETRIEVE → RERANK → FINALIZE → RETRIEVE → RERANK → FINALIZE`,
  ending `no_verified_candidate_in_scope` (expected).
- Corrective B: two attempts on `0093` and `0098`, both ended no-match; no recovery.
- C corrective attempts: 0 on the 10-task subset.
- C 0042 P15 rejected the unknown-brand claim; the shadow verifier also rejected it.

## 6. Cost

### A/B full 30 tasks

- A: 99 Planner calls, 33 search calls, 147,750 input / 9,055 output tokens,
  373.1 s agent latency.
- B: 95 Planner calls, 35 search calls, 139,410 input / 8,695 output tokens,
  359.1 s agent latency.
- Both: 0 hard-constraint violations.

### C 10 tasks

- Planner: 33 calls, 10 RETRIEVE actions, 51,552 input / 3,038 output tokens.
- Explain: 32,227 input / 1,662 output tokens.
- Shadow Verifier: 19,793 input / 904 output tokens; 37.3 s verifier latency.
- Agent latency: 452.6 s total (about 45 s/task); client E2E including MCP cold starts:
  1,123.3 s total (about 112 s/task, including ~20–31 s server startup/task).
- GPU: one Qwen instance, sequential execution, no second 9B model.
- Observed peak VRAM: **23,709 MiB used, 384 MiB free**, total 24,093 MiB.
- No OOM, but the margin is extremely small and is a real operational risk.

A/B direct latency (~12 s/task) is not comparable to C MCP latency (~112 s/ task including
cold start, explanation and verifier).

## 7. Human Review Sheet

Generated:

- `runs/p20_agent_eval/claim_review_sheet.json`
- `runs/p20_agent_eval/claim_review_sheet.csv`
- `runs/p20_agent_eval/claim_review_sheet.md`

Contains 17 natural Claim–Evidence rows with:

- product id;
- claim text;
- cited evidence refs and selected span/excerpt;
- P15 status/reasons;
- Qwen Verifier verdict/reason;
- blank human verdict/notes;
- `human_review_status = pending_independent_human_review`.

No label was marked `human_verified`. The 49-case Verifier checklist remains separately
available and provisional.

## 8. Regression and Git

- Relevant existing test suite: **149 passed, 1 skipped** (last run before this evaluation
  scripts-only change).
- P18 known state: **1 failed, 5 passed** (pre-existing `catalog_search.py` freeze-hash drift).
- P18 Frozen Manifest untouched; no P18 Reserved tuning.
- No production bug fix was made in this round. The MCP planner empty-action failure is
  reported as an unresolved bug rather than silently patched.
- New evaluation scripts are untracked until the final commit.
- Git baseline for this round: `079ea77`.
- No push.

## 9. Trustworthy Capability Boundary

Supported by this evaluation:

- Real Qwen task-level execution can be run on 30 natural tasks in one process.
- ASK/no-match/abstain behavior is strong on this set: 11/11 for A and B, 4/5 for C.
- Hard-constraint violations are 0 in the evaluated runs.
- Memory-conflict ASK works in the sampled tasks.
- C Final answers can contain *natural* semantic overclaims that P15 accepts; two out of
  16 P15-valid claims were not supported by the Verifier.
- Complete Shadow can add non-trivial latency and MCP startup cost; it does not change Gate,
  final answer or Memory.
- One natural MCP run failed with a Planner empty-action error.

Not supported by this evaluation:

- Human-verified quality;
- Corrective RAG improving natural task success (it did not here);
- initial-retrieval-miss recovery;
- production reliability under concurrent users;
- safe automatic interception or answer rewriting;
- representative C4/metadata-conflict claim coverage in the C subset.

## 10. Highest-Value Improvements

1. **Fix the Planner empty-action / MCP error path**
   - Natural C failure: `unknown planner action ''`.
   - Needs a bounded, observable retry or clear fallback; do not silently convert to success.

2. **Unify memory loading between direct A/B and MCP evaluation**
   - `0049` failed in direct A/B and succeeded in MCP C; task success is not currently
     comparable across harnesses.
   - One shared task evaluator should be the next measurement fix.

3. **Strengthen natural claim support checks**
   - P15 accepted the polyester-material overclaim from a field label (`material`) and the
     wireless-earbud preference overclaim.
   - These are better handled by the existing verifier/guards or a small evidence-value
     check, not by more prompt expansion.
