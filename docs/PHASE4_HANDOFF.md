# AgentRec-X — Phase 4 Handoff

Compressed context for a fresh conversation. Source of truth: the repository, Git history, tests
and `docs/ARCHITECTURE.md`. This file states **where things stand and what must not be reopened**;
it deliberately carries no implementation history.

---

## 1. Current State

```
branch        master
HEAD          69f56ef  milestone17: calibrate the evaluator and measure Prompt v2 (Phase 3.1)
              (+ this handoff commit)
parent        570068c  milestone16: real DeepSeek evaluation with record and replay (Phase 3)
origin/master 6845e78  (10 commits behind; nothing has been pushed)
worktree      clean
```

Completed milestones, in order:

| Milestone | Commit | What it added |
|---|---|---|
| Phase 1 — model-driven `AgentPolicy` | `6db2386` | `StructuredModelClient` seam, `LLMAgentPolicy`, provider adapter shape, scripted double |
| Phase 2 — constraint enforcement | `e5d1e47` | `CandidateEligibilityEvaluator`, `FeasibleCandidateView`, feasibility projection, guard awareness |
| Phase 2.1 — run-scoped candidate reference boundary | `86ad127` | `PolicyContext.grounded_candidates`, executor-level run-membership enforcement, model target selection |
| Phase 3 — real DeepSeek + record/replay | `570068c` | live stdlib transport, `RecordingModelClient` (live/record/replay), usage/latency/cost, `live_model` variant |
| Phase 3.1 — evaluator calibration + Prompt v2 | `69f56ef` | measurement fix, `PROMPT_V2`, parse protocol v3, `read_signature` observability |

Two independent control planes share one renderer: the accepted DAG (`recommendation/agent/`) and
the 2.0-alpha bounded loop (`recommendation/control/`). Both finalize through
`finalize_stage`. Phase 1–3.1 work lives in the control plane; the accepted DAG is unchanged.

---

## 2. Architectural Invariants

Do not weaken these. Each is enforced by a type or a component, not by convention.

1. **The policy proposes; trusted code decides.** `ActionProposal != ValidatedAction`. The model
   chooses *what action to propose*; `ActionValidator`, the capabilities, `ResultVerifier`,
   `CompletionGuard` and `LoopController` own legality, execution, budgets and termination.
2. **Only candidate-producing actions introduce identity.** `RECOMMEND_FROM_HISTORY`,
   `SEARCH_CATALOG`, `FIND_SIMILAR`, `SELECT_SOURCE` — through trusted sources and
   `GroundingVerifier`. No policy can add a candidate.
3. **Reasoning actions are read-only.** `GET_DETAILS`, `COMPARE`, `TRADE_OFF`,
   `CHECK_COMPATIBILITY`, `BUNDLE`, `VERIFY` read facts and cannot change membership, reorder the
   ledger, or approve a completion. They consume no tool-call budget.
4. **`CandidateLedger` preserves provenance.** Append-only, keyed by `parent_asin`,
   multi-source, no removal API. Exclusion from a recommendation never deletes an entry.
5. **Reasoning targets must belong to the current run.** `ReasoningExecutor._require_run_membership`
   checks every named identity against the run's live grounded allowlist **before** any catalogue
   read, and fails the whole action (all-or-nothing) otherwise. Catalogue membership authorizes
   *grounding*, never *attention*.
6. **`FeasibleCandidateView` controls recommendation membership.** A proved `VIOLATED` constraint
   makes a candidate `INELIGIBLE`; it stays in the ledger and stays inspectable, but cannot be
   presented. The projection is applied at finalization; the retrieved set is preserved.
7. **`UNKNOWN` is neither PASS nor FAIL.** It is `UNRESOLVED`: excluded from the verified feasible
   set, never reported as a violation, never silently satisfied. A constraint dimension the
   catalogue cannot check maps to `ConstraintKind.UNVERIFIABLE` rather than being dropped.
8. **Memory commit remains validated.** Only `MemoryWriteProposal` + `MemoryProposalValidator`;
   the policy has no write path.
9. **FINISH remains guarded.** A proposal carries no completion signal; `CompletionGuard`
   authorises it and may refuse a claim of compliance nothing supports.
10. **All loops remain bounded.** Step, tool-call and retry budgets are controller-owned; the
    reasoning path re-enters the step-budget check.

Policy identity is recorded, not assumed: every trajectory step carries `policy_name` and
`execution_mode` (`live` / `record` / `replay` / `scripted`), so a run cannot be filed under a
label it did not run under.

---

## 3. Current Measured Results

Verified at `69f56ef`:

| | Value |
|---|---|
| Offline test suite | **1817 passed, 33 skipped** (`pytest -q`, no network, no credential) |
| `compileall` | clean |
| `tests/test_docs.py` | 63 passed |
| `tests/test_phase31_calibration.py` | 19 passed |
| `tests/test_provider_record_replay.py` | 78 passed |
| Evaluation cases | 14 |

Real DeepSeek (`deepseek-flash`), same 14 cases, catalogue, tools, budgets, parser, runtime and
evaluator — **only the prompt differs**:

| Metric | Prompt v1 (corrected) | Prompt v2 |
|---|---|---|
| **Cases passed** | **4 / 14** | **9 / 14** |
| Clarifications | 9 | 4 |
| Repeated read-only occurrences | 18 | 7 |
| Budget exhausted | 2 | 1 |
| Premature FINISH | 0 | 0 |
| Model calls | 64 | 54 |
| Tokens in / out / total | 80 799 / 3 296 / 84 095 | 80 333 / 2 833 / 83 166 |
| Model latency (total) | 59 409 ms | 50 232 ms |
| Trajectory wall latency | 525 ms | 562 ms |
| Estimated cost | unknown (no pricing configured) | unknown |
| Failures by component | `completion: 9, policy_selection: 2, trajectory_efficiency: 1` | `completion: 4, policy_selection: 1, trajectory_efficiency: 1` |

Six cases changed verdict under v2 — **all six improved, none regressed**. No aggregate score is
produced anywhere; every dimension is reported separately and attributed to a component.

**Remaining failures under v2 (5):**

| Case | Terminal | Note |
|---|---|---|
| `novel-product-need` | `asked_clarification` | repeats the same `search_catalog`, then asks |
| `missing-requirement` | `asked_clarification` | asks correctly, but also ran a forbidden action |
| `premature-finish-recovery` | `asked_clarification` | never reaches FINISH, so guard-refusal recovery is untested against a real model |
| `empty-source-recovery-required` | `asked_clarification` | same shape as above |
| `complex-multi-step` | `budget_exhausted` | only residual repetition (3 reads) |

**Record/replay status: working, and offline-verified.**

* v1 recording replays 4/14 with **62 replayed calls, 0 live, 0 network**.
* v2 recording replays 9/14 with **53 replayed calls, 0 live, 0 network**.
* Replay reproduces executed actions, terminal outcome, failure dimensions, constraint verdicts,
  attribution and per-case model-call counts. Archives are replayed under the contract and parse
  protocol they were recorded with (`--prompt v1|v2`, `--parse-protocol`, `--prompt-version`).
* Version constants: prompt **v2**, parse protocol **v3**, recording format **1**, evaluation plane
  **4**, constraint eligibility **1**. Recordings default to `/tmp/agentrecx-recordings/` —
  outside version control on purpose.
* Live commands require explicit opt-in (`--mode live|record --confirm-live`); `pytest` needs no
  network or credential.

---

## 4. Remaining Gaps

Only currently unresolved gaps:

1. **Trusted evidence ceiling for unresolved constraints.** The catalogue carries no attribute
   that can decide some declared constraints, so the honest outcome is `UNRESOLVED` and an empty
   feasible set. `unresolved-constraint` demonstrates this on every run.
2. **No real `SimilarItem` index.** `FIND_SIMILAR` exists as an action and argument contract with
   no real item–item similarity behind it.
3. **Residual repeated / no-progress reasoning.** 7 occurrences across 4 cases under v2
   (`complex-multi-step` still terminates on budget). Reduced by the prompt change, not eliminated.
4. **No public RecSys benchmark.** The 14 cases are inspectable fixtures over a 4-item synthetic
   catalogue. Single provider, small sample, no rate claims.
5. **No production packaging.** In-process, CPU-only, offline; no service, container or deployment.
6. **Planner is NOT currently justified.** No trajectory shows a reachable action sequence the
   one-action-per-step loop could not produce. Residual failures are wrong *action choice*, not a
   failure to decompose.

---

## 5. Phase 4 Motivation

The Phase-3/3.1 evidence points at one capability ceiling, and it is an evidence ceiling rather
than a reasoning one.

* **Some `UNKNOWN` constraints cannot be resolved because the trusted catalogue simply lacks the
  fact.** `unresolved-constraint` declares a constraint dimension the catalogue cannot check for
  any candidate. The model inspects, reads and then finishes; the guard refuses the completion
  because nothing is proved compliant, and the run ends in an honest refusal. That is the runtime
  behaving correctly against evidence that does not exist. No prompt can change it, and inventing
  the fact would be fabrication.
* **Repeated `GET_DETAILS` cannot create new evidence.** Under v1 the model issued twelve
  consecutive `get_details` on unchanged targets until its step budget ran out. Prompt v2 reduced
  this to a two-observation exchange that terminated correctly on the refusal — which shows the
  loop is bounded and the contract can discourage the pattern, but *also* that reading the same
  source again has no path to a different answer.
* **Therefore the next capability should let the agent obtain new grounded evidence and/or widen
  retrieval** — a real similarity source, and a trusted path to evidence that is not already in the
  run's grounded set. Anything added must go through the existing boundaries: a new *trusted
  candidate-producing or evidence-producing* capability, never a widening of what a reasoning
  action may read, and never a model-authored fact.

Phase 4 should be scoped from these measured failures, not from a general ambition to add
capability.

---

## 6. Do Not Reopen

Do not redesign any completed phase unless Phase 4 exposes a **real regression** with a failing
test to prove it. In particular, do not revisit the evaluator's terminal semantics, the
`FeasibleCandidateView` rules, the run-scoped reference check, the recorder's modes or the prompt
contract without measurement to justify it.

Do not add:

* Planner / subgoals / reflection
* multi-agent architecture
* RL / SFT / reward model
* commerce (cart, checkout, payment, order)
* production deployment
* public benchmark
* a `NoProgressGuard` (remaining evidence is too weak; re-measure after Phase 4)

Do not change benchmark cases to improve a model's score, and do not tune a prompt against
individual cases. Preserve both recorded baselines and keep `RuleBasedPolicy`,
`scripted_model` and `live_model` distinguishable.

**Phase 3.1 principle, still in force:** fix measurement before optimising behaviour, then change
one policy variable at a time.
