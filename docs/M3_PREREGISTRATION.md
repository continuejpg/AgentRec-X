# M3 PREREGISTRATION — live Agent policy vs accepted fixed fusion

**Status: FROZEN for everything except the LLM provider identity (see §9).**
**No M3 run may start, and no M3 result may be inspected, until §9 is filled in.**

This document is a preregistration, not a report. It exists so that the one remaining
unmeasured claim in this project is decided by a protocol fixed in advance, and so that a
result — whichever way it goes — cannot be re-interpreted, re-tuned or quietly re-run.

---

## 1. The single question

> Under the same frozen candidate sources, the same 20 000-user cohort and the same shared
> evaluator, does the **actual Agent policy** behave differently from the **fixed fusion
> baseline**?

This is the project's central unmeasured claim. M2 recorded it as `EVIDENCE MISSING`; M3 exists
to replace that with a measured paired result.

Explicitly **not** in scope: anything that would change the system in order to move the number.
No prompt tuning, no routing/selection-policy tuning, no tool-budget tuning, no fusion change,
no weights, no depth, no new candidate sources, no model training, no new baseline search.

---

## 2. Frozen evaluation protocol

| item | frozen value |
|---|---|
| cohort | the frozen 20 000-user cohort — **used in full; no reduction, no sampling, no re-draw** |
| cohort seed / selection | `cohort_seed = 20260201`, `cohort_size = 20000`, deterministic seeded permutation stratified by history length (`eligible_users = 412445`) |
| catalogue | 156 746 items, PAD row 0 excluded |
| split | `temporal_leave_two_out` |
| evaluator | `recommendation.evaluation.batched.evaluate_batched` reached via `experiments.benchmark_public.evaluate_arm` — **unmodified, and not forked for this evaluation** |
| K values | 5, 10, 20 |
| batch size | 256 |
| cohort identity check | the run must assert its target identities are identical to the frozen baseline evidence before scoring anything |

## 3. Arms

**Comparator — `fixed_fusion` (accepted).** The accepted Phase-5 arm, reconstructed
deterministically from frozen source heads; identity is pinned by SHA256 in §8. It fuses
`FUSION_SOURCES = (popularity, sequential, metadata)` with `HEAD = 1000`, `RRF_K = 60`,
equal weights and order-preserving first-occurrence dedup. It must not be re-tuned, re-weighted
or re-depth-ed for M3.

**Treatment — the actual Agent policy.** `recommendation/control/model_policy.py::LLMAgentPolicy`
driving the existing bounded control plane (`recommendation/control/loop.py`) over the existing
`CandidatePlane`, `CandidateLedger`, `GroundingVerifier` and constraint machinery, with their
current semantics. A model-driven policy reaches its decisions through a live
`StructuredModelClient` built by `recommendation/control/provider_adapter.py::build_provider_client`.

**Frozen budget — `LoopLimits` defaults, not adjusted for M3:**

| limit | value |
|---|---|
| `max_steps` | 6 |
| `max_tool_calls` | 4 |
| `max_retries` | 1 |

The model-call backstop implied by these (`per_step = 12`) is `12·(max_steps+1) + 2·(max_retries+1) + 32 = 120` calls per user.

**Prohibited substitutions.** The Agent arm must not be driven by `ScriptedModelClient` or any
other test double, and no "simplified Agent" may be written for this evaluation. Using a scripted
client would measure the script, not the policy. The TIGER candidate source stays in the state
M2/M1 recorded: `QUALIFIED`, **not enabled in the runtime**; M3 does not enable it.

**Optional diagnostic arm (not the conclusion).** The existing `agent_selected` **rule proxy**
(`arm_agent_selected` + `agent_selection_plan`) may be reported as a diagnostic, because it is now
cheaply reconstructible from the same frozen heads. It is a deterministic rule, **not** the Agent
policy, and no conclusion about the Agent may rest on it.

## 4. Endpoints

- **Primary endpoint: NDCG@10**, paired per user, Agent − fixed_fusion.
- **Secondary:** Recall@5/10/20, HR@5/10/20, NDCG@5/20.
- **Behavioural (Agent arm only):** mean/median tool calls per user; tool-selection distribution;
  source-selection distribution; fallback count and rate; failure/refusal count and rate;
  termination reason distribution; mean steps; provider-error and timeout counts; wall time.
- **Paired outcome accounting:** gained users (comparator miss → Agent hit), lost users
  (comparator hit → Agent miss), both-hit, both-miss — at each K.
- Correctness/provenance/determinism claims are **not** given intervals; they are pass/fail facts.

## 5. Statistics (fixed in advance)

The Step-2.8b protocol is reused unchanged: `SEED = 20260201`, `RESAMPLES = 10 000`.

- **Recall/NDCG paired deltas** → paired user-level percentile bootstrap over the 20 000 users
  resampled with replacement; report the observed paired mean difference and the 2.5th/97.5th
  percentiles as a 95% CI. For this single-positive protocol Recall@K ≡ hit@K.
- **hit@K differences** → exact paired McNemar, no continuity correction:
  `two-sided p = min(1, 2·Σ_{i≤min(b,c)} C(n,i) / 2^n)`.
- Report per user-level sign stability across K.

**Classification rule, declared before any result exists** (identical to M2's, so the two
milestones stay comparable):

| class | condition |
|---|---|
| `SUPPORTED` | at every K: McNemar p < 0.05 **and** both the Recall@K and NDCG@K bootstrap CIs exclude 0 |
| `DIRECTIONAL ONLY` | point estimates share one sign across K, but the `SUPPORTED` conditions are not all met |
| `NO DETECTABLE DIFFERENCE` | point estimates do not share a sign across K |
| `EVIDENCE MISSING` | no usable per-user evidence |

**Wording rule.** Any CI that crosses 0 forces `DIRECTIONAL ONLY`. No
deterministic-superiority language is permitted for such a result. `SUPPORTED` is **not** a claim
of practical superiority; it only means the difference is distinguishable from 0 at this cohort
size.

## 6. Stop rule and anti-tuning rule

- **Stop after one execution of the protocol, whatever the sign.** A negative result — the Agent
  losing to fixed fusion, or no detectable difference — is an acceptable and publishable M3
  outcome.
- After results are seen, it is **forbidden** to change: the prompt, the routing/selection policy,
  the tool budget or `LoopLimits`, the fusion configuration, the candidate sources, the cohort,
  the K values, or the statistical protocol; and forbidden to re-run the Agent arm in the hope of
  a better draw.
- The Agent arm is executed **once**, in a single frozen run over the whole cohort, with its
  provider configuration recorded in provenance. Re-execution for the purpose of selecting a
  favourable draw is prohibited. Any retry is internal to `LoopLimits` and must be reported in the
  behavioural statistics.
- A behaviour log sufficient to explain what the Agent actually called, per user, is a
  **completion condition**, not a nice-to-have.

## 7. What counts as M3 COMPLETE

1. A complete paired result for actual Agent vs fixed fusion on the frozen cohort.
2. Uncertainty reported per §5, with the predeclared classification applied.
3. Behaviour logs sufficient to explain what the Agent invoked.
4. Project thesis wording updated to match the measured outcome.

## 8. Frozen comparator evidence (this milestone's replay)

The `fixed_fusion` comparator is materialised per user. Produced by
`runs/m3/m3_replay_popularity_metadata.py` (committed reproducibility code), which was authorised
to deterministically replay **only** the already-accepted `popularity` and `metadata` arms.

**The large NPZ artifacts deliberately stay gitignored** — they are recorded here by SHA256 and
live under `runs/m3_evidence/`, in the same `VERIFY-ONLY` durability class as the other large
artifacts. They are not committed.

| artifact | bytes | SHA256 |
|---|---|---|
| `runs/m3_evidence/phase5_popularity_hits.npz` | 2 141 746 | `f7719c1dc843472a8f66dd2eb788cc3dc155e638a56b522b425501be8e4ad44c` |
| `runs/m3_evidence/phase5_metadata_hits.npz` | 2 141 746 | `22cdf563cd9fc651a8bd1a0ce7b0158c8df49a69f4c7fe8121a1b13165822096` |
| `runs/m3_evidence/phase5_fixed_fusion_hits.npz` | 2 141 746 | `62c2bc7c8013dfd651e4773e6dc328ae645e49f42af59d71fa421834a464b23f` |
| `runs/m3_evidence/phase5_fusion_source_heads.npz` | 76 410 797 | `c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4` |
| `runs/m3_evidence/M3_REPLAY_PROVENANCE.json` | 5 151 | `891cd0999f1f7ed09dc027872981fbd4a894b8ec4433b0f5bc67dfa48bc7806c` |

**Replay provenance (summary).**

- cohort 20 000/20 000, catalogue 156 746, `K_VALUES=(5,10,20)`, `batch_size=256`, benchmark version 1;
  the run asserted its cohort target identities are byte-identical to the frozen baseline evidence.
- inputs, by SHA256: sequences `d3d83426…ab81`, mappings `dca7815a…1912`, products `175c83ca…71dd`,
  accepted aggregates `runs/phase5_benchmark_public/arms.json` `66b591e2…e5da`, frozen sequential
  head cache `/tmp/step28_heads_cache.npz` `d0361df7…90dc`.
- the SASRec head was **not recomputed**: the frozen Step-2.8 head cache was hash-verified and its
  top-10 proven identical to the frozen Phase-5 `sequential` head for all 20 000 users.
- reproduction result: replay aggregates matched the accepted Phase-5 aggregates exactly
  (tolerance 1e-12 for rate metrics, 5e-5 for the 4-dp `mean_target_rank`):
  popularity R@5/10/20 = 74/105/157, metadata 101/185/297, fixed_fusion 172/287/465,
  `mean_target_rank = 80590.2987`.
- determinism: round-tripping the frozen heads file through `arm_fixed_fusion` +
  `evaluate_batched` reproduced `target_rank` identically for all 20 000 users, with max
  |Recall delta| = 0 and max |NDCG delta| = 0 against the accepted values.
- shape fact that any rebuild must respect: the `popularity` and `sequential` heads are 1000 deep,
  but the **`metadata` heads are ragged** (min 99, max 496, mean 456.53; no user reaches 1000,
  none empty), so the artifact stores padded matrices plus explicit `*_lengths` arrays.
- verification note: an unmasked `fused_head(top_n=K)` rebuild does **not** reproduce hit@K,
  because the accepted pipeline applies seen-history masking before taking top-K. The comparator
  must always be rebuilt through the shared evaluator, never by truncating a raw fused head.

## 9. OPEN ITEM — the only thing not yet frozen

**The LLM provider and model identity are NOT frozen.** They must be supplied and recorded before
any Agent execution, and they must not be selected, changed or re-tried after seeing any M3 result.

To be filled in before the run:

| field | source | frozen value |
|---|---|---|
| base URL | `AGENTRECX_LLM_BASE_URL` | *OPEN* |
| model | `AGENTRECX_LLM_MODEL` | *OPEN* |
| credential | `AGENTRECX_LLM_API_KEY` | *supplied out-of-band; never logged, stored or printed* |
| sampling parameters (temperature, seed, max tokens) if configurable | provider config | *OPEN — must be recorded and held constant* |

Freezing these is a **preregistration amendment**, recorded in this file before execution. An
amendment that changes anything in §2–§6 after a result has been seen invalidates M3.

**Feasibility warning carried forward.** With `LoopLimits` as frozen, one run over 20 000 users is
at least ~20 000 provider calls and up to ~2.4 M. The cohort must not be reduced to fit a budget
(§2). If cost makes the full cohort infeasible, that is a preregistration change to be negotiated
**before** any result is seen — not a result-driven adjustment.
