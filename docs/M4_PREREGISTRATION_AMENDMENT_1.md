# M4 PREREGISTRATION AMENDMENT 1 — expose the frozen source values in the action contract

**Amendment version:** 1
**Date:** 2026-09-29
**Status:** FROZEN before the formal execution.

This amendment does **not** overwrite or rewrite the preregistration. The original document stays
byte-identical at `docs/M4_PREREGISTRATION.md`, and the amendment that follows is the public record
of a defect found during preflight and the exact repair applied.

---

## 1. Hashes

| document | SHA256 |
|---|---|
| original preregistration, `docs/M4_PREREGISTRATION.md` | `7a18ea7258a3a175227a270ee38177c58bde66eadd51af8ccb02c91f83b981af` |
| this amendment, `docs/M4_PREREGISTRATION_AMENDMENT_1.md` | recorded in `docs/M4_EXECUTION_MANIFEST.json` after generation |

The original hash is unchanged and is still asserted by `assert_preregistration_hash()`. The
amendment is hashed separately and recorded in the execution manifest, so a run is bound to **both**
documents.

## 2. Confirmation that the formal cohort had not run

**The formal 20,000-user M4 cohort has NOT been started.** `runs/m4_execution/` does not exist, no
formal provider calls were made, and no M4 recommendation metric of any kind had been computed
before this amendment. The only adaptive-arm execution to date is the 12-user preflight described in
§4, which is an environment check, not a result.

## 3. The defect

`build_action_schema` (`recommendation/control/model_policy.py:205-238`) renders each argument of an
offered action as exactly three fields:

```json
{"name": "source", "type": "string", "required": true}
```

The trusted argument model already restricts `source` to
:class:`~recommendation.control.arguments.CandidateSource` members, but that restriction was never
made **visible**: the provider-facing schema carried no `enum` and no value list. The provider
therefore had no way to learn the four legal names and invented one instead.

M4's action-space projection (preregistration §7.3) narrowed the offered actions to
`SELECT_SOURCE` and `FINISH` and passed the schema through unchanged, so the omission reached the
provider intact.

## 4. Evidence

Every probe and every adaptive step produced `"default"`, which is not a `CandidateSource` member:

| observation | result |
|---|---|
| 5 direct provider calls with the projected schema | `{"source": "default"}` in **5 / 5**, 0 valid values |
| projected schema's `source` argument | `{"name":"source","type":"string","required":true}` — no enum |
| 12-user frozen-order two-arm preflight, **fixed** arm | 48 tool calls, 0 failures, 0 empty rankings |
| 12-user frozen-order two-arm preflight, **adaptive** arm | 2 tool calls, **11 failures**, **10 empty rankings** |

Mechanism: `"default"` fails `ActionProposal` construction as `invalid_arguments`; the policy retries
once under `max_attempts`, then raises `PolicyActionError`; the loop terminates the run
`ABORTED / NO_AVAILABLE_ACTION` with an empty ledger, which is scored as an empty ranking.

**The 12-user adaptive preflight is invalid as an efficacy result** and must not be quoted as one:
it measured the broken contract, not source-selection ability. It is retained only as the
before-fix evidence above.

## 5. The repair

The four frozen source names are exposed to the provider as a closed value set, **M4-scoped**:

```text
history · catalog_search · similar_item · two_tower
```

Implemented in `experiments/m4_schema_projection.py` — the injected client that already constructs a
replacement `ModelRequest` — as `annotate_source_values`. The projected `source` argument gains
`"enum"` and `"allowed_values"`, each containing exactly those four strings. The annotation is
applied to copies, so the shared builder's output is not mutated.

Nothing about which sources may be selected changes: the provider chooses among the same four
sources it was always meant to choose among. Only the visibility of the already-valid set changed.
The guard in `experiments/m4_policies.py` continues to refuse any source outside the universe as a
`protocol_deviation`, so the closed universe is enforced in two places rather than one.

### Why this is not a broader change

| bound | status |
|---|---|
| shared control-plane behaviour | **unchanged** — `build_action_schema` still emits name/type/required; asserted by a test |
| source universe | unchanged — the same four sources |
| action semantics | unchanged |
| cohort, order, materialised heads | unchanged |
| provider, model, temperature, thinking, JSON mode, `max_tokens` | unchanged |
| policy logic, guard, tool budget | unchanged |
| ranking, evaluator, metrics, statistics | unchanged |
| completion rules | unchanged |

## 6. What is unchanged in the science

The research question, the frozen cohort and its order, the four-source universe, the materialised
heads, the per-source depth, the fusion, the provider configuration, the adaptive policy, the tool
budget, the evaluator, the metrics and the paired statistics are **all unchanged**. This amendment
corrects the action *contract's* completeness; it does not alter the experiment.

## 7. Retained deviation from the concurrency probe

The non-cohort provider concurrency probe consumed **19 calls against a 16-call budget** (3 over),
because a confirmation run at the selected 4-way level was counted after the fact rather than
budgeted beforehand. The deviation is recorded in `docs/M4_EXECUTION_MANIFEST.json` and in the
commit message rather than being quietly reconciled. It affected only the probe, not the cohort, and
the probe's selection (4-way: zero failures, zero 429s, full schema compliance at every level)
stands.

## 8. Gate for proceeding

The formal run proceeds only if a fresh 12-user frozen-order adaptive preflight passes: 12/12 users
complete, zero invalid-source failures, zero source-naming `PolicyActionError`s, every emitted
source one of the four enum values, no retries caused by invalid source strings, non-empty rankings
except where a source legitimately returned nothing, and trajectories inside the frozen policy and
tool budget. That gate is a functionality check only, and no prompt or policy is tuned in response
to recommendation quality.

## 9. Execution record (appended after the run; the text above was not altered)

This section records what happened. Nothing above it was edited, so the amendment's hash covers the
frozen text and this appendix together, and the manifest records that combined hash.

**A second defect was found and repaired before execution.** Once the enum repair was in place the
adaptive arm still failed 12 of 12 preflight users. The cause was not the schema but the guard: a
duplicate-source proposal raised `ProtocolDeviation`, which subclasses `RuntimeError` rather than
`PolicyActionError`, so the loop's deviation handling did not catch it and the run terminated
`execution_failed` with no ranking at all. That contradicted the frozen deviation rule, which
requires the run to terminate **at once on the ranking it already holds** and keeps the user in the
primary analysis. The guard now records the deviation and substitutes `FINISH`; recording a
deviation and being terminated were also separated as two states, so the duplicate path does not
misreport itself as a hard termination. This repairs the deviation *mechanism*, not the deviation
*rule*: the duplicate still never executes and never reaches the ledger.

**Gate result.** The fresh 12-user frozen-order adaptive preflight passed. Fixed: 12/12 users, 0
failures, 0 empty rankings, 48 tool calls, 60 steps. Adaptive: 12/12 users, 0 failures, 0 empty
rankings, 0 retries, 0 protocol deviations, 23 tool calls, 35 steps, emitting only legal source
values (`history` 12, `two_tower` 11). Functionality only; no quality-based tuning followed.

**Execution.** The formal 20,000-user cohort was executed **exactly once** at the validated 4-way
provider concurrency (see §7 for the probe's retained 19-call-versus-16-call deviation). Cohort
identity `e14356b2843835741fd652fbe1b845ebfe951bf968ebba673fb239fe9a325e56`; all 20,000 users were
run in both arms and none was excluded. The recorded result is `docs/M4_RESULT.md`; the artifact it
describes is `runs/m4_formal_result.json`, SHA256
`f99ac820d9d7853279d0f611d9a68e66fd77b7bfbbbd0f7f068e00f4d0f352af` — gitignored per AGENTS.md §8,
which is exactly why its contents are transcribed into the committed result document.

**No result-driven change was made.** No prompt, policy, source universe, tool budget, ranking,
evaluator or metric was altered after any result was seen, and the cohort was not re-run.
