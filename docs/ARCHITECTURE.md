# AgentRec-X — Current Architecture

> **Status: CURRENT.** This is the architecture of the worktree on
> `feat/agent-source-limit-ablation` after the adaptive preference construction change.
> Authority order: **code** → this document → historical documents. If a historical document
> disagrees with current code, this document wins.

AgentRec-X is a recommendation system with a bounded Agent layer. The recommendation
backbone is deterministic; the optional LLM layer may propose a route or a supported
preference, but it cannot write candidate identity, scores, catalogue evidence, ranking or
metrics.

---

## 1. Architectural goals

1. **Truth stays in deterministic code.** Identity, retrieval order, fusion, catalogue
   facts, constraints, final order and metrics are produced by trusted components.
2. **The Agent may decide what to ask, not what is true.** A model plan is optional and
   schema-bounded; LLM output never becomes a product row, score, or piece of evidence.
3. **No silent fallback.** A missing model or optional ranker is reported through
   `/health` or an error, never disguised as a successful recommendation.
4. **Preferences are scoped and conservative.** Explicit memory is user-owned; task
   preferences exist only inside a shopping task and are never written to long-term memory.
5. **Evaluation claims are layered.** Formal offline evaluation, fixed-pool experiments,
   constructed scenarios and engineering validation are reported separately.
6. **Reproducibility is part of the product boundary.** Temporal splits, frozen cohorts,
   deterministic tie-breaking and recorded artifacts precede quality claims.

---

## 2. Serving V1 and request surfaces

### Serving V1 — the current default

The default profile is `serving_v1` (`AGENTRECX_FUSION_PROFILE` unset or set to that name):

```text
user message + trusted history
        |
        v
SASRec sequential  +  Two-Tower retrieval  +  Catalogue Lexical (BM25)
        \                 |                       /
         +----------------+----------------------+
                          |
                 reciprocal_rank_fusion
                 RRF_K = 60 · source head = 100
                          |
                 catalogue evidence / titles
                          |
                 preference / task soft rerank
                          |
                 final recommendations + trace
```

The three sources produce different quantities, so only their **rank lists** are fused.
The fused score is labelled `rrf_rank_fusion`; provenance remembers each source's rank and
score kind. `PAD = 0`, seen-history items and unknown identities never enter the result.

`/health` reports `requested_engine`, `active_engine`, `fallback_used`, `fallback_reason`,
the fusion sources and the TIGER state. If Serving V1 cannot be built, the deployment may
fall back to SASRec-only and says so over HTTP; it does not silently claim fusion.

### Current surfaces

| Surface | Code path | State | Purpose |
| --- | --- | --- | --- |
| `POST /v1/recommend` | `recommendation/api/app.py` → `RecommendationEngine` | stateless | Direct top-k recommendation from a caller-supplied trusted history |
| `POST /v1/demo/sessions` | `recommendation/api/demo_routes.py` → `DemoSessionManager` | server session | Open an isolated demo shopping session |
| `POST /v1/demo/sessions/{session_id}/chat` | `AgentGraph` + `RecommendationTool` | session | The bounded conversational graph path |
| `POST /v1/demo/sessions/{session_id}/agent/shopping` | `DemoAgentService` → `LoopController` | session | Multi-turn shopping Agent with research, clarifications and adaptive preferences |
| `POST /v1/demo/agent/recommend` | `DemoAgentService.run` | stateless | One bounded Agent request with explicit trusted history |
| `GET /demo/` | `recommendation/web` | browser | Plain HTML/CSS/JS demo |
| `GET /health`, `GET /v1/model` | `recommendation/api/app.py` | stateless | Readiness and model metadata |

`/v1/recommend` is **not** the shopping Agent. It is the direct model-service API: it takes
history, runs the configured engine, and returns ranked candidates. The multi-turn shopping
path adds session state, intent maintenance, candidate research, clarification/recovery,
preference construction and a bounded loop; it uses the same trusted recommendation tool and
engine underneath.

---

## 3. Component map

| Area | Modules | Owns | Does not own |
| --- | --- | --- | --- |
| Sequential retrieval | `recommendation/inference/sasrec.py`, `recommendation/models/`, `recommendation/training/` | SASRec checkpoint loading, history encoding, scored candidates | HTTP, Agent policy, preference interpretation |
| Two-Tower retrieval | `recommendation/inference/twotower.py` | Item-id history → cosine-ranked candidates | Identity resolution, fusion |
| Lexical retrieval | `recommendation/inference/lexical.py` | Whole-catalogue BM25 ranking | Catalogue facts for final cards |
| Fusion | `recommendation/inference/fusion.py`, `recommendation/control/candidate_ledger.py` | Fixed source profiles, RRF, provenance, depth policy | Learned reranking |
| Optional ranker | `recommendation/inference/learned_ranker.py` | LightGBM LambdaRank over the fused legal pool, with counted RRF fallback | Default serving path, quality claims |
| Optional TIGER | `recommendation/inference/tiger_engine.py`, `backends/tiger_public/` | External generative backend boundary and serving adapter | Default fusion, public quality claims |
| Catalogue | `recommendation/catalog/` | `parent_asin`-keyed metadata index and normalization | Retrieval |
| Product research | `recommendation/control/product_research.py` | Inspect/compare/list a bounded set of products from trusted records | Free-form web browsing, fabricated facts |
| Agent control | `recommendation/control/` | `ActionProposal`/validator, `LoopController`, `CandidatePlane`, ledger, observations, grounding, verification, constraints | Making up candidates or scores |
| Shopping intent | `recommendation/demo/shopping_intent.py` | Hard/soft intent, conflicts, task-scoped preference entries | Long-term memory policy |
| Shopping service | `recommendation/demo/agent_service.py` | Per-session composition, research/clarification/recovery, task preferences, response projection | Model internals, database ownership |
| Adaptive preferences | `recommendation/demo/preference_construction.py` | Candidate evidence, counterfactual Top-K, interaction choice, task soft reranking | Long-term preference writes |
| Memory | `recommendation/memory/` | Explicit preference store, extraction, lifecycle | Task-only choices |
| Matching/reranking | `recommendation/preference_matching/`, `recommendation/reranking/` | MATCH / VIOLATION / UNKNOWN evidence and deterministic order | Candidate generation |
| HTTP/session | `recommendation/api/`, `recommendation/demo/sessions.py` | Route validation, session isolation, locks, error mapping | Ranking truth |
| Evaluation | `recommendation/evaluation/`, `experiments/` | Temporal split, masking, metrics, frozen arms | Serving policy |
| Browser | `recommendation/web/` | Display, option buttons, transcript | Any ranking or evidence invention |

---

## 4. Recommendation and Agent flows

### 4.1 `/v1/recommend` — direct recommendation

```text
POST /v1/recommend
  -> validate request/history
  -> RecommendationEngine.recommend(history, k)
  -> SASRec-only or Serving V1 fusion
  -> response + model metadata
```

There is no session, Agent policy, clarification or preference question on this path.
Callers must supply trusted history explicitly; the service does not infer it from text.

### 4.2 `/chat` — bounded graph path

```text
session -> AgentGraph
  -> deterministic decision model or configured LLM plan
  -> RecommendationTool
  -> ProductEnricher
  -> PreferenceCandidateMatcher
  -> PreferenceReranker
  -> ChatResponse + trace
```

This is the older bounded graph contract. It is retained for the chat endpoint and its
tests; the richer shopping surface below composes the control-plane loop.

### 4.3 `/agent/shopping` — current multi-turn shopping Agent

```text
AgentShoppingRequest + session
  -> DemoAgentService.run_shopping_turn
  -> derive ShoppingIntent from trusted snapshot + current message
  -> plan candidate sources (trusted history, catalogue search, optional similar/two-tower)
  -> LoopController: policy -> validated ActionProposal -> CandidatePlane
  -> observations / grounding / eligibility / constraint checks
  -> optional dynamic product research, comparison, clarification, recovery
  -> final candidates after hard constraints
  -> optional adaptive preference question (max two rounds)
  -> user answer -> soft rerank of the frozen candidate window
  -> AgentRecommendResponse (recommendations, intent, research, trace, outcome)
```

Important boundaries:

* **Hard constraints are applied before soft preferences.** A task preference can reorder the
  already-eligible candidate set; it cannot add a candidate that failed price, weight or
  category constraints.
* **Task state is separate from long-term memory.** `task_key` owns pending questions and
  task preferences; `user_key` owns the accepted explicit preference memory. Several demo
  sessions may share a `user_key`; they do not share task preferences.
* **A clarification is a suspension, not a successful answer.** A pending preference or
  research question returns `waiting_for_user`; the next valid answer resumes the same task.
* **A failed run is not repaired by prose.** The response carries a terminal failure or a
  question; it never substitutes a plausible list.

### 4.4 Optional control-plane policy

The same `DemoAgentService` can use a provider-backed policy for the bounded stateless
`/agent/recommend` path and for LLM-selected tool plans. The model returns a typed
`ActionProposal`; `ActionValidator` checks the action, arguments and allowed candidate
references before `LoopController` executes anything. There is no free-form tool execution.

---

## 5. Adaptive preference construction

The adaptive preference path is deterministic and candidate-aware. It is not a learned
policy and not an information-gain estimator.

1. **Trusted evidence.** `ProductResearchTool.compare_products` reads catalogue facts for at
   most five presented candidates. A dimension is eligible only when every candidate has a
   comparable value; missing values are not filled in.
2. **Unresolved dimensions.** Explicit, confirmed and revoked attributes are removed, as are
   hard-constraint dimensions and already-known task preferences. The remaining dimensions
   are ranked by contextual relevance, coverage, normalized difference and counterfactual
   impact.
3. **Counterfactual Top-K.** Each executable option is simulated against the current
   preference state. An option survives only if it changes the current preference window's
   order. The result records `baseline_top_k` and `counterfactual` predictions.
4. **Interaction or skip.** The policy returns `ASK_DIRECT`, `SHOW_CONTRAST`, `EXPLAIN` or
   `SKIP`. `EXPLAIN` requires a prior confirmed task choice and contextual relevance; a bare
   first question uses direct/contrast mode. A maximum of two preference rounds is allowed.

When the user answers:

* the chosen option becomes a task-scoped `TaskPreference` (`confirmed=true`, source
  `user_selected`); `none`/`skip` creates no preference;
* the service applies the preference to the **same frozen candidate response** that produced
  the question, so before/expected/actual use one Top-K window;
* the final ordering is recomputed after hard filtering, and reasons are rebuilt from the
  final intent;
* `preference_outcome` reports `changed` and `verified`; a non-change is reported honestly;
* no speculative or task-scoped choice is written to long-term memory.

The second round is allowed only after a substantive first answer and only when another
eligible option still changes that window. The newest confirmed preference is the primary
refinement; older confirmed preferences remain tie-breakers.

---

## 6. Defaults and optional components

| Component | Default | How it is enabled | Status |
| --- | --- | --- | --- |
| SASRec | on for the serving path | artifact configuration | current |
| Two-Tower | on in `serving_v1` when its artifact is present | `AGENTRECX_FUSION_PROFILE` / artifact presence | current source |
| Catalogue BM25 | on in `serving_v1` | artifact configuration | current source |
| RRF | on in `serving_v1` | fixed `RRF_K=60`, head 100 | current |
| Learned Ranker | off | `AGENTRECX_FUSION_PROFILE=serving_v1_learned_ranker` + model/schema/LightGBM | experimental; not claimed better than RRF |
| TIGER | off | `AGENTRECX_TIGER=1` + backend assets | optional/qualified; not default |
| LLM Agent policy | off | provider configuration + explicit mode/flag | optional proposal layer |
| Shopping Agent loop | on for `/agent/shopping` | session endpoint | current |

The current default is deterministic and offline. No provider credential is required for the
default demo. The optional LLM Agent mode is a proposal layer only.

---

## 7. Evaluation boundaries

| Layer | Entry points | What it answers |
| --- | --- | --- |
| **Formal offline evaluation** | `recommendation/evaluation/`, `experiments/benchmark_public.py`, `experiments/benchmark_twotower.py`, frozen protocol docs | Full-catalogue, temporal leave-two-out ranking quality on the frozen cohort; shared evaluator and masking |
| **Fixed-candidate / fusion experiments** | `experiments/fusion_controls.py`, B1 evidence artifacts, pre-registered reports | Whether a source or fusion arm improves the same cohort under one protocol |
| **Constructed shopping scenarios** | `experiments/p3_scenario_eval.py`, `p4_multi_turn_eval.py`, P6 research/recovery evaluations | Constraint adherence, tool use, recovery and preference behaviour on constructed cases |
| **Engineering validation** | `tests/`, startup/readiness scripts, browser acceptance | Contract correctness, fixture wiring, API behavior, UI wiring and failure modes |

These layers are not interchangeable. A constructed scenario pass is not an online shopping
success rate. A fixed-pool result is not a full-catalogue result. No A/B test, click data,
conversion or user-satisfaction experiment exists in this repository.

`runs/` is for local/generated experiment output and is ignored by default; selected small
frozen evidence files are tracked explicitly. Large checkpoints, processed data and memories
are not committed.

---

## 8. Trust boundaries

```text
[ UNTRUSTED / CONVERSATIONAL ]        may influence:
  user message                         which supported preferences are stored
  optional LLM plan                    whether the Agent asks a question
                                        presentation order through confirmed preferences
      does not own: identity, scores, catalogue facts, evidence, rank, metrics

[ TRUSTED DETERMINISTIC CODE ]         owns:
  trusted history                      parent_asin <-> item_id · PAD excluded
  catalogue facts                      MetadataIndex / ProductResearchTool
  retrieval and fusion                 fixed sources · RRF_K=60
  evidence                             MATCH / VIOLATION / UNKNOWN
  constraints                          hard filtering before soft rank
  final ordering                       deterministic preference and policy keys
  evaluation metrics                   shared evaluator
```

The LLM request schemas use `extra="forbid"` and have no field that could carry candidate
identity, scores, evidence, ranking or a memory mutation result. A provider failure returns a
stable error; it never becomes a partial mutation or a deterministic-looking answer.

---

## 9. Failure behaviour

* **Model/engine unavailable:** `/health` reports `model_loaded=false` or a fallback reason;
  recommendation endpoints return a typed error.
* **Fusion collaborator missing:** Serving V1 either fails loudly or falls back to SASRec-only
  with `fallback_used=true` and a reason, depending on configuration.
* **Optional ranker failure:** counted fallback to full-precision RRF on the same legal pool;
  no second recall, no silent score relabeling.
* **Tool/action error:** `LoopController` terminates with a typed reason; no fabricated
  candidate or partial success.
* **Research failure:** the response records the failure/degradation and may ask a bounded
  clarification; it does not invent a fact.
* **Invalid preference option:** the question remains pending; a wrong option is rejected
  without consuming the task.
* **Expired/duplicate/cross-session clarification:** `404`/`410`/validation error as
  appropriate; no other task is consumed.
* **No candidates:** an empty list is a valid honest result; no padding with fabricated items.

---

## 10. Determinism and reproducibility

* Temporal leave-two-out; no random split.
* `PAD=0`; canonical `parent_asin` identity; seen-item masking.
* Deterministic source ranking and RRF tie-breaking on identity.
* Frozen cohort seed `20260201` and recorded artifact digests.
* Preference decisions are deterministic; the same evidence/state yields the same action.
* Response views exclude timing and machine-specific paths from reproducibility checks.
* `AGENTRECX_*` configuration is documented in [Usage](USAGE.md) and `.env.example`; secrets
  are supplied at runtime and never committed.

---

## 11. Directory map

```text
recommendation/
  models/ training/ datasets/     SASRec training and data path
  inference/                      SASRec, Two-Tower, BM25, fusion, optional LR/TIGER
  control/                        bounded Agent loop, action validation, candidate plane,
                                  ledger, grounding, verification, product research
  agent/                          legacy bounded graph used by /chat
  demo/                           sessions, shopping intent, Agent service, adaptive
                                  preferences, runtime composition, schemas
  memory/                          explicit preference memory (SQLite)
  preference_matching/             MATCH / VIOLATION / UNKNOWN evidence
  reranking/                       deterministic preference reranking
  catalog/ rag/                    metadata index and candidate-scoped enrichment
  api/                             FastAPI app and demo routes
  web/                             plain browser UI
  evaluation/                      temporal split, masking, metrics
  baselines/                       ItemCF engineering baseline
  semantic_id/                     historical GenRec v0
  backends/                        external backend boundary
experiments/                       formal benchmarks and exploratory experiments
scripts/                           setup, start, artifact recovery
config/                            artifact manifest and runtime configuration
docs/                              current docs, historical records, frozen reports
tests/                             synthetic deterministic test suite
runs/                              local experiment output; selected evidence tracked
data/                              raw/processed artifacts (ignored)
```

Every package under `recommendation/` has a README describing its local contract. The current
documentation index is [`docs/README.md`](README.md); the historical architecture is
[`HISTORICAL_ARCHITECTURE.md`](HISTORICAL_ARCHITECTURE.md).

---

## 12. Known limitations

* Offline evaluation only; no live inventory, pricing, cart, CTR/CVR or online A/B claims.
* No preference-conditioned relevance labels; no satisfaction or user-quality claim.
* The catalogue lexical source is the main current latency cost.
* Learned Ranker is experimental, artifact-dependent and is not claimed to beat RRF.
* TIGER is optional/qualified and not in the default serving path.
* The shopping Agent is a bounded, local demo; sessions and pending questions are in-process,
  and there is no authentication or durable identity system.
* Adaptive preferences are limited to trusted comparable attributes (`price`, `weight`,
  `color`, `material`, `brand`) and at most five comparison candidates per question.
* Constructed scenario results are not production shopping-success metrics.
