# AgentRec-X — Architecture

This document explains **what each part of the system owns and where its boundaries are**.
It is deliberately not API-reference documentation: per-package READMEs next to the code
hold the detailed contracts, and this file holds the shape.

Cross-references: [README](../README.md) · [Experiments](EXPERIMENTS.md) ·
[Usage](USAGE.md) · [Project history](PROJECT_HISTORY.md).

---

## 0. Canonical diagrams (start here)

These two diagrams are the **canonical shape** of the repository. Everything below this section
is the detailed reference; where the two disagree, the diagrams are the summary and the numbered
sections are the detail.

They describe **two stacks that live in one repository but are not one identical runtime stack**:
the browser serving path, and the offline research/evaluation path.

---

### Diagram A — Serving / interview architecture

The path the verified browser demo actually takes. Only components on that path appear here.

```text
User / Browser
      |  same-origin HTML/CSS/JS, no build step, no CDN, no hosted LLM, no API key
      v
FastAPI app  (recommendation/api/app.py)   .   browser UI (recommendation/web)
      |
      v
DemoSessionManager  (recommendation/demo/sessions.py)
      |  session_id, user_key, per-session preference-memory namespace
      v
AgentGraph  (recommendation/agent/graph.py)
      |  conversational orchestration; decision seam is deterministic and offline
      v
RecommendationTool  (recommendation/tools)
      |
      v
SASRecInferenceEngine  (recommendation/inference/sasrec.py)
      |  the accepted checkpoint - the ONE candidate generator on this path
      v
Trusted candidate set
      |  canonical identity parent_asin <-> item_id; PAD = 0 is never a candidate
      |
      +-------------------------------+---------------------------------+
      |                                                                 |
      v                                                                 v
MetadataIndex + ProductEnricher                          PreferenceMemoryService
(recommendation/catalog, recommendation/rag)             (recommendation/memory)
catalogue facts, candidate-scoped evidence               SQLite, per user_key
                                                         explicit preferences (ADD / REPLACE / REMOVE)
      |                                                                 |
      +-------------------------------+---------------------------------+
                                      v
                     PreferenceCandidateMatcher  (recommendation/preference_matching)
                     per-preference evidence: MATCH / VIOLATION / UNKNOWN
                                      |
                                      v
                     PreferenceReranker  (recommendation/reranking)
                     deterministic lexicographic policy
                     (fewer violations, then more matches, then original order)
                                      |
                                      v
                     Grounded, preference-aware recommendation
                     original SASRec rank preserved alongside the final rank
                                      |
                                      v
                     Response serialization  (recommendation/demo/serialization.py)
                                      |
                     +----------------+----------------+
                     v                                 v
              ChatResponse (JSON)          Recommendation Trace   <-- OBSERVABILITY ONLY
                                           a view over the response's own state
                                           never an input to ranking
```

**Authority split on this path.** The conversational layer and the truth owners are different
layers, and the boundary is structural rather than conventional:

```text
===================== TRUST BOUNDARY =====================

[ CONVERSATIONAL / DECISION SIDE ]  may influence:
    - which supported preferences are stored
    - whether the recommendation flow is invoked (route)
    - presentation order, through supported preferences

    cannot write to anything on the truth side below

----------------------------------------------------------

[ TRUSTED CODE ]  owns and decides:
    - the user's behavioural history      (application-owned, read-only to the agent)
    - canonical product identity          (parent_asin <-> item_id; PAD = 0 excluded)
    - catalogue facts                     (MetadataIndex)
    - recommendation model output         (SASRec scores)
    - evidence truth                      (PreferenceCandidateMatcher)
    - the final preference-aware order    (PreferenceReranker)
    - evaluation metrics                  (shared evaluator)

==========================================================
```

**Decision authority is narrower than truth authority.**

### Component ownership

| Responsibility | Owner |
| --- | --- |
| Behavioural history | trusted application state |
| Candidate generation | SASRec (`SASRecInferenceEngine`) |
| Product facts | catalogue metadata (`MetadataIndex`) |
| Explicit preferences | `PreferenceMemoryService` |
| Evidence | `PreferenceCandidateMatcher` (deterministic) |
| Final preference-aware order | `PreferenceReranker` (deterministic) |
| Conversational routing | `AgentGraph` decision seam |
| Observability | `Recommendation Trace` (read-only view) |

**Deliberately not on this path:** the candidate plane, the candidate ledger, the bounded
control plane, Two-Tower, TIGER and the fusion comparator. The browser agent consults exactly one
candidate generator (SASRec) and performs no adaptive multi-source routing.

---

## 0b. Diagram B — Offline research / evaluation architecture

Everything here is **offline**. Nothing in this diagram runs inside the browser demo.

```text
Frozen dataset + protocol
  Amazon Reviews 2023 . Sports & Outdoors . chronological temporal leave-two-out
  frozen 20,000-user cohort (deterministic seed) . full catalogue . seen-item masking
                              |
     +--------------+---------+---------+-------------------+
     |              |                   |                   |
     v              v                   v                   v
  SASRec       Two-Tower            TIGER-FP32        popularity and
  (also the    (dual encoder)       (Semantic-ID      metadata retrieval
   serving     RESEARCH ONLY        generative)       (lexical / non-personalised)
   model)                          RESEARCH ONLY      RESEARCH-ONLY ARMS
     +--------------+-------------------+-------------------+
                    |
        Offline benchmark arms  (experiments/, recommendation/evaluation)
                    |
          +---------+---------+
          v                   v
    Single-source         Fixed fusion
    evaluation            (rank fusion over popularity + sequential + metadata)
          +---------+---------+
                    v
          Shared evaluator  - one implementation for every arm
                    |
          Recall / HR / NDCG at K = 5/10/20
                    |
                    v
          Paired uncertainty  (exact McNemar . paired user bootstrap)
                    |
                    v
          Research conclusions and frozen reports  (docs/reports/)
```

A separate, later experiment sits alongside the benchmark arms and is **not** part of the browser
serving path:

```text
Bounded LLM control plane  (recommendation/control - a SECOND orchestration stack)
                    |
                    v
  existing real Agent system  (LLMAgentPolicy + LoopController, bounded step/tool budget)
                    |
                    vs
                    |
  fixed-fusion comparator  (same cohort, same evaluator, same protocol)
                    |
                    v
  Paired evaluation on the frozen 20,000-user cohort
                    |
                    v
  Result: the bounded policy underperformed the comparator on the primary endpoint.
  Behavioural reason: 19,999 of 20,000 trajectories were `recommend_from_history -> finish`;
  catalogue search was never invoked, source selection was unavailable in that frozen
  configuration, and the policy returned four candidates.

  This is a result about ONE system configuration under a frozen protocol.
  It is NOT a claim that LLM agents do not work for recommendation.
```

---

## 0c. How the two diagrams relate

The **serving stack** is the interview/demo product surface. It is what `/demo/` runs.

The **research stack** evaluates alternative retrievers, fusion strategies and a bounded LLM
policy under the same frozen recommendation protocol, using the same dataset, cohort and
evaluator.

Research findings can inform design decisions, but most research sources are deliberately **not**
wired into the browser serving path: the serving path uses SASRec and the original conversational
stack only.

The repository therefore contains **two orchestration stacks** — the original browser `AgentGraph`
stack and the later bounded control-plane stack. They share data, protocol and provenance
conventions; they are not interchangeable, and they are not one identical serving architecture.

---

## 1. Architectural goals

1. **One trained model, everything else inspectable.** SASRec is the only learned component.
   Every other decision — retrieval, matching, ordering, routing — is deterministic and
   reviewable by reading the code.
2. **The conversational layer must not be able to corrupt behavioural data.** This is the
   primary safety property, and it is enforced structurally rather than by convention.
3. **Never fabricate.** Missing metadata stays missing; unavailable evidence stays
   `UNKNOWN`; a failed stage raises instead of degrading into plausible-looking output.
4. **No silent quality claims.** Offline policy diagnostics measure adherence to a
   configured policy, not relevance. The evaluation protocol, the benchmark and the
   diagnostics are reported as three separate categories of evidence.
5. **Compose, do not duplicate.** Each milestone added a stage behind an existing seam, and
   later stages reuse earlier contracts instead of re-implementing them.
6. **Reproducibility as a feature.** Deterministic seeds, recorded digests, frozen
   protocols, and deterministic payloads that exclude timing and machine metadata.

---

## 2. Component map

| Module | Owns | Does not own |
| --- | --- | --- |
| `recommendation/preprocess.py`, `io_utils.py` | raw review ingestion, k-core filtering, deterministic id mapping, processed artifacts | any modelling |
| `recommendation/datasets/` | SASRec training dataset, negative sampling, inference-time history encoding | scoring |
| `recommendation/models/` | SASRec architecture | training loops |
| `recommendation/training/` | deterministic trainer, checkpoint I/O, selection state | data splitting |
| `recommendation/evaluation/` | the frozen temporal leave-two-out protocol, masking, metrics | any ranking policy for serving |
| `recommendation/baselines/` | ItemCF engineering baseline | benchmark claims |
| `recommendation/inference/` | checkpoint loading, catalog scoring, deterministic top-k ranking | HTTP, agent concerns |
| `recommendation/tools/` | `RecommendationTool`: the agent-facing recommendation contract | model internals |
| `recommendation/catalog/` | `parent_asin`-keyed metadata: normalization, artifact, lookup | retrieval, ranking |
| `recommendation/rag/` | candidate-scoped evidence retrieval (`ProductEnricher`) | candidate generation |
| `recommendation/memory/` | explicit preference memory: schemas, stores, lifecycle (`PreferenceMemoryService`) | extraction from history, ranking |
| `recommendation/preference_matching/` | three-state preference↔metadata evidence (`PreferenceCandidateMatcher`) | ordering |
| `recommendation/reranking/` | the frozen ordering policy (`PreferenceReranker`) and its offline evaluation | evidence semantics |
| `recommendation/agent/` | `AgentGraph` orchestration, state, decision contract | any recommendation logic |
| `recommendation/api/` | FastAPI app, M6 endpoints, demo routes, error mapping | recommendation logic |
| `recommendation/demo/` | demo profiles, `DemoSessionManager`, `DemoRuntime`, public serialization | recommendation logic |
| `recommendation/web/` | browser assets (plain HTML/CSS/JS) | any client-side ranking |

---

## 3. Recommendation backbone

```text
raw Amazon Reviews 2023  ->  preprocessing (iterative k-core, deterministic ids)
                         ->  processed sequences + mappings artifacts
                         ->  SASRec dataset (temporal leave-two-out)
                         ->  SASRec training (validation-selected checkpoint)
                         ->  sealed test evaluation (once)
                         ->  accepted checkpoint + run manifest
```

Invariants that later layers depend on:

* interactions are **chronologically ordered**; there is no random splitting anywhere;
* item id **0 is PAD** and is never a real item; real ids start at 1;
* `parent_asin` is the **canonical** product identity and is treated as an **opaque
  string** (nothing assumes a `B` prefix);
* the accepted mappings stay authoritative for `parent_asin <-> item_id`; no later layer
  redefines identity;
* ranking is a **total order**: higher score first, then lower item id.

---

## 4. Recommendation Tool boundary

`recommendation/tools/RecommendationTool` is the *only* interface through which anything
above the model layer obtains candidates.

* `RecommendationToolRequest` carries the model-facing argument (`k`) and nothing else.
* `RecommendationContext` carries **trusted** interaction history and is passed as a
  separate argument, so history arrives from application state rather than from model or
  conversation output.
* The engine is injected once and reused; the Tool never loads a checkpoint.
* Domain errors are a stable taxonomy (`MissingUserHistory`, `UnknownHistoryItem`,
  `InvalidRecommendationRequest`, `RecommendationUnavailable`), and the Tool normalises
  engine failures instead of leaking framework internals.
* Score semantics are explicit: the value is a **raw ranking score**, not a probability,
  confidence, rating or preference score.

The Tool is a hard boundary, not a convenience wrapper: `AgentGraph` cannot reach SASRec by
any other path, so the graph cannot score, mask or rank candidates itself.

---

## 5. AgentGraph

`recommendation/agent/AgentGraph` composes the accepted stages as **optional injected
collaborators**. The graph owns orchestration only.

Node set (superset; each node exists only when its collaborator was injected):

```text
load_memory -> decide -> { direct -> finalize
                         | recommend -> enrich -> match_preferences -> rerank -> finalize }
            -> persist_memory
```

`AGENT_GRAPH_VERSION = 2` (bumped by Milestone 10D when the two preference nodes were
added).

| Injected | Route | Milestone behaviour |
| --- | --- | --- |
| `tool` | `decide -> {finalize \| recommend -> finalize}` | M7B / M7C |
| `+ product_enricher` | adds `enrich` | M8 |
| `+ memory_service` | adds `load_memory`, `persist_memory` | M9 |
| `+ preference_matcher` + `reranker` | adds `match_preferences`, `rerank` | M10D |

Construction-time validation is explicit: matcher and reranker must be supplied
**together**, a matcher requires an enricher (M10A reads the metadata M8 attached), and
each collaborator must expose its one method. A half-configured preference stage fails
loudly instead of being silently skipped. The graph never constructs a default matcher,
reranker, metadata store or memory store.

Collaborators are declared as `runtime_checkable` `Protocol`s (`ProductEnricherLike`,
`PreferenceMemoryLike`, `PreferenceMatcherLike`, `PreferenceRerankerLike`), so
`recommendation/agent` imports none of `rag`, `memory`, `preference_matching` or
`reranking`. AST guards in the test suite enforce that.

**Decision seam.** The decision model receives exactly
`build_decision_messages(user_message)` and returns an `AgentDecision` validated against a
frozen schema (`extra="forbid"`), which is why a decision cannot smuggle interaction
history. The formal tests and the demo inject a deterministic offline rule; no provider SDK
is imported anywhere.

**Presentation.** `finalize` chooses its rendering branch by *which structured stage ran*
(a `RerankingReport`, else an enrichment result, else the raw Tool result) — never by
inspecting a string. When reranking is present the response is rendered in reranked order
while keeping `original_rank` auditable.

---

## 6. Control plane (AgentRec-X 2.0-alpha, IMPLEMENTED)

The accepted `AgentGraph` (section 5) is a fixed DAG: one routing decision per run, and the
sequence after it is static. `recommendation/control/` adds a second, **opt-in** control
plane that decides the next action from an observation - a bounded agent loop. Its founding
rule is that this changes *who decides the next step* and nothing about *how a
recommendation is made*.

```text
PolicyContext --(AgentPolicy.choose)--> ActionProposal          untrusted
                                              |
                                    ActionValidator              controller-owned
                                              |
                                        ValidatedAction          stamped metadata
                                              |
                                  Trusted ActionExecutor         accepted pipeline
                                              |
                                    DomainResult                 NOT policy-visible
                                              |
                                     ResultVerifier              identity / counts / ranks
                                              |
                                      ObservationAdapter         minimised
                                              |
                                      Observation                what the policy may see
                                              |
                                       LoopController            owns the loop
```

Loop topology (`recommendation/control/topology.py`):

```text
initialize -> check_limits -> policy -> validate_action -> dispatch
                   ^                                         |-- execute -> verify
                   |                                         |             -> observe
                   |                                         |             -> update_state
                   +-----------------------------------------+----------------+
                   `-- complete -> finalize / refuse / abort -> END
```

Nodes commit their routing through LangGraph `Command(goto=...)`, because a conditional edge
is resolved from the state *before* its node runs and therefore cannot see a verdict
produced inside that node.

**Authority split.**

| Component | Owns | Must never do |
| --- | --- | --- |
| `AgentPolicy` | `choose(PolicyContext) -> ActionProposal` | execute a tool, write state, count steps, touch trusted history, supply candidate ids, commit memory, validate its own output, mark the run finished |
| `LoopController` | lifecycle, counters, `available_actions`, dispatch, budgets, termination, trajectory | score, rank, retrieve, interpret preferences, commit memory, render |
| `ActionValidator` | availability, argument re-validation, stamping `action_id` / `step_index` / `run_id` / `turn_id` | repair a proposal into a different action |
| `ResultVerifier` | Tool grounding, count and rank integrity, candidate identity, stage alignment | mutate, repair or fabricate a result |
| `CompletionGuard` | whether `FINISH` is a legal end | certify a failed or ungrounded run |

**Boundaries held as types**, not conventions: `ActionProposal != ValidatedAction`,
`PolicyContext != AgentGraphState`, `DomainResult != Observation`, and `FINISH != finished`.
A proposal has no field for history, a tool name, SQL or execution metadata; a
`PolicyContext` has no field for trusted history, a memory key, a store, an engine or
candidate ids; a `DomainResult` never reaches the policy.

**Bounded execution.** `max_steps`, `max_tool_calls` and `max_retries` are controller
configuration the policy cannot read or raise. Exhaustion terminates deterministically - the
controller does not ask the policy whether it would like to continue. LangGraph's own
`recursion_limit` is derived from the same budgets as a framework-level backstop.

**Two drivers, one engine.** `LoopController(driver="graph")` runs the compiled cycle;
`driver="direct"` runs the identical engine methods in Python, so a difference between them
is a topology bug rather than a semantics bug.

**Shared with the DAG.** `recommendation/agent/rendering.py` holds the presentation layer and
`recommendation/agent/pipeline.py` the memory-commit and finalize stages, so both control
planes render a run with one code path.

**Enabling it.** The demo runs the accepted DAG by default:

```text
AGENTRECX_CONTROL_PLANE=graph   # default: the accepted M7B-10D DAG
AGENTRECX_CONTROL_PLANE=loop    # 2.0-alpha bounded agent loop
```

---

## 7. Candidate plane (Stage 2, IMPLEMENTED)

Stage 1 allowed one candidate source. Stage 2 allows several, and this is the boundary that
keeps that safe:

```text
AgentPolicy                     chooses WHICH source - nothing else
    |
CandidatePlane                  executes that source, grounds it, records it
    |-- history        -> accepted RecommendationTool -> SASRec
    |-- catalog_search -> CatalogSearchSource over the trusted catalogue
    `-- similar_item   -> registered tool, or ABSENT
    |
GroundingVerifier               confirms every identity against trusted data
    |
CandidateLedger                 deduplicates identity, retains ALL provenance
```

**One action consults one source.** Adaptive retrieval is the point: a policy that wants two
sources proposes two actions, sees the first observation, and may change its mind. There is
deliberately no "call every source" method, because that would make fixed fan-out the easy
path and adaptive retrieval the awkward one. The fast path survives - a single
`RECOMMEND_FROM_HISTORY` action is one source and one tool call, exactly as in Stage 1.

**`CandidateLedger`** is the run's record of candidate identity and provenance.

* *Deduplicate identity, not evidence.* One entry per `parent_asin`; a later sighting
  appends a `CandidateProvenance` record rather than replacing the earlier one.
* *Never fuse heterogeneous raw scores.* `HISTORY` reports SASRec logits, `CATALOG_SEARCH`
  reports BM25, `SIMILAR_ITEM` reports similarity. Each is stored under its own `score_kind`
  and never summed or averaged. Cross-source ordering uses **reciprocal rank fusion** over
  each source's own ranking - well-defined precisely because it uses only the *order* every
  retriever can be trusted to mean.
* *Original source ranks stay recoverable*, so a candidate's full provenance can be
  reconstructed after the fact.
* Candidates may only be added by `record_from_source`, which requires a declared trusted
  source. A policy is nowhere in that call path.

**`GroundingVerifier`** answers one question: is this identity one the trusted side knows? It
checks the trusted item mapping and, when configured, the catalogue as well
(`checks_catalog` reports which). It never scores, never selects, and reads no user data. It
**fails closed**: an identity it cannot confirm is `UNGROUNDED`, retained purely as audit
evidence so a drifting source is *visible*, and never presented as a candidate.

**`CatalogSearchSource`** searches the whole catalogue lexically. It is deliberately separate
from the Milestone 8 retriever rather than a flag on it: that retriever is *candidate-scoped*
by design, which is exactly why RAG cannot widen a candidate set. Widening the scope is a
capability a policy may select, and it happens only here. The tokenizer, BM25 constants and
idf function are imported from `recommendation.rag.retrieval`, so the two retrievers cannot
drift apart in how they score text; document scope and output type are what differ. A
zero-hit search returns zero hits with `matched=False` - never an arbitrary catalogue slice.

**Unavailable sources are absent, not faked.** `SIMILAR_ITEM` is not configured by default.

### Not implemented (documented, not claimed)

* **No item-item similarity index ships.** `SIMILAR_ITEM` needs a separate build step over
  `Sports_and_Outdoors_sequences.json` (item co-occurrence) or item embeddings; neither is
  generated today. The seam is implemented and tested with a registered tool, but the source
  is unavailable unless one is explicitly registered.
* **No reranking over the fused ledger.** The accepted M10B preference reranker still runs
  over the history recommender's candidates only. Ledger-order reranking is not implemented.
* **Stages 3, 4 and 6 are not implemented.** No clarification, no task/session state, no
  memory-write proposal pipeline, no reasoning capabilities (`GET_DETAILS`, `COMPARE`,
  `TRADE_OFF`, `CHECK_COMPATIBILITY`, `BUNDLE`, `VERIFY`) and no evaluation plane. Their
  action enum members and argument contracts are declared so they can be added without
  reshaping the protocol, but no capability executes them and a proposal naming one is
  refused.

---

## 8. Interaction and personalization plane (Stage 3, IMPLEMENTED)

### Three signal domains, kept apart

The stage's central requirement is that user information is not collapsed into one profile
string. Three domains exist and are owned by different components:

| Domain | Owner | Mutability | Reaches the policy as |
| --- | --- | --- | --- |
| **A. Behavioural preference** | the host application | never written by the agent | a boolean (`has_trusted_history`) |
| **B. Long-term explicit preference** | accepted `PreferenceMemoryService` | only via a validated proposal | attributed constraint text |
| **C. Current task / session intent** | `TaskState` (this stage) | mutable during the interaction | attributed constraint text |

Domain A is application-owned and untouchable by the agent; that invariant is unchanged from
Milestone 7A.

### Task state vs persistent memory

Long-term memory deliberately takes effect **from the next turn** — the accepted Milestone 9
semantics that stops a turn from appearing to have let its own statement influence the
candidates it returned. But "not the red one *this time*" is about the current request, and
waiting a turn to honour it would be wrong. So the same sentence can produce two things:

```text
"not red this time"   -> TaskState constraint (applies now), no memory write
"never red again"     -> TaskState constraint (applies now)
                      + MemoryWriteProposal (committed for future tasks)
```

`TaskState` is mutable during the interaction, never persisted, and **immediately usable in
the turn that produced it**. A temporary override *suppresses* within the task and leaves
stored state untouched: suppression is structurally not removal, because task constraints and
memory entries are different objects.

`PreferenceScope` (`SESSION` / `TASK` / `LONG_TERM`) carries that decision, and
`PreferenceLifecycle` (`EXPLICIT` / `INFERRED_HYPOTHESIS` / `CONFIRMED` / `RETRACTED`) carries
what kind of signal it is. Only `EXPLICIT` and `CONFIRMED` may become durable preferences:
**an inference is evidence, never truth.**

### Clarification as a first-class action

`ASK_CLARIFICATION` suspends the run. It:

* runs no tool and consumes no tool-call budget;
* ends the run in `WAITING_FOR_USER` with `TerminationReason.AWAITING_USER` — neither
  `FINISHED` nor `ABORTED`, and `CompletionGuard` is never consulted;
* records a `ClarificationObservation` carrying the question, what it blocks and the offered
  options, so the trajectory can explain why the run stopped mid-way;
* is **resumable**: the next turn continues the same task by handing the suspended
  `TaskState` back, and the answer reaches the policy as a task constraint;
* is offered only while the task has not already asked, so one run cannot become an
  interrogation.

Whether clarification is *warranted* is a policy decision, not a framework rule. A policy
asks when the missing information would materially change the candidate set, ranking or
feasibility; a policy that never proposes it never enters the clarification branch.

### The memory write pipeline

```text
user text
  -> Preference Interpreter   (proposes; the accepted rule-based extractor works unchanged)
  -> MemoryWriteProposal      (value/scope/lifecycle/supporting span/turn provenance)
  -> MemoryValidator          (grounding, attribution, scope, lifecycle)
  -> PreferenceMemoryService  (the accepted, already-trusted writer) -> commit / reject
```

**A proposal is not a commit.** Nothing in `recommendation/control/memory_proposal.py` can
write the store: it holds no store handle and no `user_key`. Only a proposal the validator
marked *approved* becomes a `PreferenceCandidate` for the accepted service.

**An inference is not truth.** An `INFERRED_HYPOTHESIS` is validated as *soft evidence*
(`SOFT_ONLY`) and refused a durable commit, so an inferred long-term preference cannot
quietly become hard truth. It may still inform the current task through `TaskState`.

Validation refusals carry stable codes: `missing_value`, `removal_carries_value`,
`avoidance_expressed_as_removal`, `ungrounded_span`, `span_not_in_turn`,
`ephemeral_scope_not_persisted`, `inferred_signal_is_soft_evidence`.

The `avoidance_expressed_as_removal` check encodes the stage's explicit warning: **a negative
preference is not a memory deletion.** "I don't like red" is an `ADD` of an `AVOID`-polarity
value; only an explicit retraction is a `REMOVE`. The validator refuses to treat one as the
other.

### What the policy is shown

`PolicyContext` gained a bounded projection, not the store:

```text
task_intent                 a small vocabulary; UNKNOWN is a legitimate value
task_constraints            active, task-relevant constraint *descriptions*
hard_constraint_count       how many define feasibility
inferred_constraint_count   how many are soft evidence only
awaiting_user               whether the task is suspended
```

Relevant-memory retrieval here is **evidence selection, not mutation authority**: it selects
and bounds, and writes nothing. There is no `memory_id`, no `user_key`, no supersession chain
and no store handle in the projection, and `PolicyContext.summary()` records only counts — so
the trajectory cannot become a copy of the user's preferences.

### Not implemented (documented, not claimed)

* **No conflict-resolution algorithm across domains.** Scope and lifecycle *precedence rules*
  are represented (`hard_constraints()` vs `soft_constraints()`), but nothing yet reconciles
  a genuine contradiction between a task override and a stored preference by proposing
  `ASK_CLARIFICATION`. That belongs with Stage 4.
* **No LLM interpreter.** The accepted rule-based extractor is the only interpreter; the
  proposal seam is what a future provider adapter would implement.
* **Stages 4 and 6 are not implemented** — no reasoning capabilities and no evaluation plan.

---

## 9. Reasoning plane (Stage 4, IMPLEMENTED)

> Reasoning may change a decision; reasoning may not change a fact.

`recommendation/control/grounded_reasoning.py` reads only normalized catalogue records.
Every conclusion traces back to an attribute the catalogue actually contains, and an absent
attribute yields `UNKNOWN` rather than a plausible value. No model call is involved.

What the shipped catalogue supports (measured on 20 000 records of the real artifact):

| attribute | coverage | used for |
| --- | --- | --- |
| `Item Weight` (or `Package Weight`) | 96% | deterministic weight/portability constraints |
| `Item Package Dimensions L x W x H` | 67% | deterministic dimension facts |
| `price_text` | 77% | deterministic budget constraints |
| `Color` | 74% | matching an explicit colour constraint |
| `Brand Name` | 69% | matching an explicit brand constraint |
| `Material` | 64% | comparison only |
| categories / features | 96%+ | category constraints, comparison |

**Facts.** `GroundedFacts` exposes each attribute verbatim, plus deterministic unit conversion
for the loose human strings the artifact carries (`"1.2 Kilograms"` → 1.2 kg,
`"4.49 x 4.49 x 1.5 inches"` → centimetres). An unreadable value yields `None`; the caller
reports `UNKNOWN`. "Identity not in the catalogue" (`None`) and "in the catalogue with no
attributes" (`UNKNOWN`) are different answers and are never conflated.

**Hard constraints are three-state.** `SATISFIED` / `VIOLATED` / `UNKNOWN`, where `UNKNOWN` is
neither a pass nor a failure. A constraint whose required fact is absent must not silently pass
— that would fabricate compliance — and must not automatically fail — that would fabricate a
violation. This is what lets `CompletionGuard` refuse to certify a run whose constraints were
never actually verified. Kinds: `price_max`, `price_min`, `weight_max`, `category`, `brand`,
`color`.

**Comparison** emits a row even when every value is `UNKNOWN` (the absence of a shared fact is
a finding) and reports an unsupported attribute request rather than ignoring it.

**Trade-off** resolves a stated priority (`lighter` → weight, `cheaper` → price, `rating`) onto
a catalogue attribute, orders only the products that carry the fact, and names the rest as
unknown. It never places a product it cannot ground.

**Compatibility is grounded or `UNKNOWN`.** This catalogue carries **no interface, fitment or
model-version semantics**, so a requirement naming one of those returns `UNKNOWN` no matter how
many products are supplied — the reasoner never manufactures a "these probably fit" verdict for
a check it did not perform. Requirements it *can* ground (brand, category, colour) are answered
from catalogue attributes, and a member lacking the attribute makes the whole check unresolved.

**Bundles** emit an aggregate only when every member carries the fact; otherwise the blocking
members are named and the aggregate is `UNKNOWN`. A partial sum is never presented as a total.
A bundle is a *recommendation set under aggregate constraints*: there is no cart, order or
transaction concept anywhere in the result.

The reasoner is **read-only**. It cannot add a candidate, reorder the ledger or approve a
completion, and a test asserts it exposes no such surface.

### Which identities a reasoning action may name (Phase 2.1, IMPLEMENTED)

Read-only is necessary and was not sufficient. Until Phase 2.1 the executor asked the catalogue
whether an identity existed and treated "yes" as authorisation, so a reasoning action could name
**any product the catalogue contained** even when the run had never retrieved it. Membership
never changed, but the boundary the docstrings claimed — *facts about candidates the run already
holds* — was not the boundary the code enforced.

The enforced rule is now:

```text
candidate-producing actions   may introduce identity, through trusted sources only
reasoning actions             may only REFERENCE identity the current run already holds
```

`ReasoningExecutor.execute` takes the run's live grounded allowlist and checks every named
identity against it **before** the catalogue is consulted, so an unauthorised reference never
causes a read and never produces an observation. The check is all-or-nothing: `COMPARE(A, X)`
with `X` outside the run is refused whole rather than answered partially, because a partial
answer would leak which of the named identities the run holds.

`PolicyContext.grounded_candidates` is the bounded reference projection a policy chooses
targets from — identity plus reasoning position, capped at the largest identity list any
argument model accepts. It narrows the allowlist the context already exposed and adds no
identity; a reference carries no catalogue value, so choosing a target never means reading one.

Reasoning authority is deliberately **not** recommendation membership (section 11): an
`ineligible` or `unresolved` grounded candidate remains a legal reasoning target, which is what
lets an agent explain an exclusion or gather the fact that would resolve it. Inspecting such a
candidate changes no verdict and cannot return it to the recommendation set.

### Not implemented (documented, not claimed)

* **No `CHECK_COMPATIBILITY` semantics for interfaces or fitment**, because the catalogue has
  none. The verdict is `UNKNOWN` by design; adding real fitment checks would require a
  different data source.
* **No semantic reasoning over requirements.** A requirement is matched against catalogue
  wording deterministically; there is no interpretation of an open-ended natural-language
  requirement.
* **No catalogue browsing.** A later phase might add a trusted candidate-producing lookup
  capability; reasoning actions will not become one.

---

## 10. Model-driven policy (Phase 1, IMPLEMENTED)

The bounded loop already let an Observation change the next action; the policy that made that
choice was deterministic.  This phase adds a **model-driven** policy behind the same seam, and
changes nothing else.

```text
AgentState
    |
PolicyContextBuilder            bounded projection - no history, catalogue, store or executor
    |
LLMAgentPolicy                  builds a prompt, parses the answer
    |
ActionProposal                  UNTRUSTED structured proposal
    |
ActionValidator                 owns legality: availability + declared argument model
    |
ValidatedAction -> Trusted ActionExecutor
    |
DomainResult -> ResultVerifier -> ObservationAdapter
    |
Observation -> State update
    |
LLMAgentPolicy again
```

### The authority split

| The model decides | Trusted code decides |
| --- | --- |
| which action to propose, from `available_actions` | whether that action is legal (`ActionValidator`) |
| which approved candidate source to consult | how the action executes (`CandidatePlane`, capabilities) |
| whether to ask, retrieve facts, compare, or finish | which product identities exist (`CandidateLedger`, `GroundingVerifier`) |
| a short non-authoritative rationale | which catalogue facts are true (`GroundedReasoner`) |
| | whether behavioural history holds (application) |
| | whether memory may be committed (`MemoryProposalValidator`) |
| | whether a hard constraint holds (deterministic) |
| | whether `FINISH` is permitted (`CompletionGuard`) |
| | every budget and termination (`LoopController`) |

`LLMAgentPolicy.choose(context)` returns an :class:`ActionProposal` and nothing else.  It has no
access to a tool, a ledger, a store, a catalogue or an executor, so it cannot bypass one.  A
`FINISH` proposal carries no completion signal; the guard still authorises it.

### Structured output

The model is asked for one JSON object and the answer is parsed into the repository's own
`ActionProposal`.  There is no substring matching.  Three rejection classes, kept distinct
because they mean different things:

```text
not JSON              -> a formatting failure, corrected once within the same policy call
unknown / unoffered   -> a protocol violation, refused
invalid arguments     -> the action is permitted but its arguments violate its declared model
```

The per-action argument models are the contract, and `ActionProposal.model_post_init` already
enforced them before this phase; the policy adds no coercion, so a near-miss is a rejection
rather than a repair.

### What the model is shown

A short policy-contract prompt plus a bounded JSON payload: the user request, task intent and
constraints, whether trusted history exists, a **candidate count** (not the candidates), the
previous observation *summarised* (kind, status, counts) and the remaining budget.  The offered
actions and their arguments are **generated from `ARGUMENTS_BY_ACTION`**, so the prompt cannot
describe an action that does not exist or omit one that does.

Deliberately absent, and therefore unable to leak: trusted behavioural history, the memory
`user_key`, the catalogue, the candidate ledger, raw scores, product metadata, grounded
reasoning facts, and the full trajectory.  A reasoning observation contributes *that facts were
read*, never the facts - a test asserts that a product title and weight from a detail
observation do not appear in the prompt.

### Provider independence

`StructuredModelClient` is the whole seam: `complete(request) -> ModelResponse`.  A real
provider is a small adapter
(`recommendation/control/provider_adapter.py` ships the shape for any OpenAI-compatible chat
endpoint, with an **injected** transport so no HTTP library is imported and no key is stored).
Every test injects `ScriptedModelClient`, so the suite needs no network and no credential.

### Bounded failure

`max_attempts` bounds the model calls *within one policy decision*, and one decision is one
controller step.  A model that never answers usefully therefore ends the run through
`PolicyActionError` → `NO_AVAILABLE_ACTION`, not through an unbounded loop.  A provider
exception or timeout is normalised the same way.  A single malformed answer is corrected once,
with the parse error passed back as a hint.

### Documentation of the seam, not of a provider

This phase shipped the adapter contract and no live call: a live call would have made the suite
non-reproducible.  The seam is configurable through `AGENTRECX_LLM_BASE_URL`,
`AGENTRECX_LLM_MODEL` and `AGENTRECX_LLM_API_KEY`.  Phase 3 added the transport, the recording
layer and the real-model evaluation; see **section 24**.

### Not implemented in this phase (documented, not claimed)

* **No live provider smoke was executed *in this phase*.** The adapter was tested against a stub
  transport.  A real call was made in Phase 3 (section 24).
* **No prompt tuning or evaluation against a real model *in this phase*.** The model variant was
  driven by a deterministic double, so it measured the *runtime*, not a model's competence.
  Real-model measurement is Phase 3.
* **No model-driven planning or subgoal generation** - one next action per step, as before.
* **No fine-tuning, RL or reward model.**
* **Hard-constraint narrowing was not implemented in this phase** - the model is not used to
  hide that, and the model policy cannot remove a candidate. Narrowing arrived in Phase 2
  (section 11) and is owned by trusted code, not by a policy.

---

## 11. Hard-constraint enforcement and the feasible candidate view (Phase 2, IMPLEMENTED)

`CandidateLedger != FeasibleCandidateView`
------------------------------------------
Before this phase the repository could **check** a hard constraint and could not **enforce**
one. The trusted `GroundedReasoner` already produced a per-candidate three-state verdict, but
nothing turned that verdict into task eligibility: the ledger is a provenance record with no
notion of feasibility, the candidate plane built its result from raw tool output filtered only
by grounding, and every renderer iterated whatever the candidate set held. A product the
catalogue proved violated the budget could therefore be presented as a recommendation, and the
evaluation plane could only *report* that it had been.

The two questions are now answered by two different components, and must stay apart:

| | Question | Owner |
| --- | --- | --- |
| `CandidateLedger` | did this product ever enter the candidate universe, from which source, at what rank, was its identity grounded? | `recommendation/control/candidate_ledger.py` |
| `FeasibleCandidateView` | for the constraints active **on this task, right now**, is this candidate feasible? | `recommendation/control/constraint_eligibility.py` |

```
CandidateLedger              complete audit universe - never filtered, never rewritten
      |
Constraint evaluation        GroundedReasoner.check_constraint  (deterministic, per candidate)
      |
Task eligibility             CandidateConstraintAssessment -> CandidateEligibility
      |
FeasibleCandidateView        verified_eligible / ineligible / unresolved  + full detail
      |
FeasibilityProjection        the renderer-facing artifacts, narrowed to verified feasible
      |
Ranking / finalization       accepted renderer, unchanged
```

```text
provenance is preserved; eligibility is constrained.
```

### Three-state semantics

The per-constraint verdicts are the accepted ones and are kept in full; the eligibility layer
only *derives* from them.

| Verdict | Meaning | Effect on task eligibility |
| --- | --- | --- |
| `SATISFIED` | grounded evidence proves the constraint holds | stays eligible with respect to that constraint |
| `VIOLATED` | grounded evidence proves it does not hold | candidate becomes `INELIGIBLE` |
| `UNKNOWN` | trusted evidence is insufficient | candidate becomes `UNRESOLVED` - neither pass nor fail |

```text
if ANY active hard constraint == VIOLATED:  INELIGIBLE
elif ANY == UNKNOWN:                        UNRESOLVED
else:                                       VERIFIED_ELIGIBLE
```

A proved violation **outranks** an unresolved constraint: excluding a candidate already
disproved needs no further evidence, and calling it merely unresolved would leave a known-bad
candidate inside the feasible view. `CandidateConstraintAssessment` keeps every individual
verdict, with its observed value, so "why was this product excluded" is answerable from the
trajectory rather than inferred from a boolean.

### Enforcement is not the model's

`CandidateEligibilityEvaluator` takes a `GroundedReasoner` and nothing else - no policy, no
model, no memory. It is stateless. A model policy and `RuleBasedPolicy` therefore receive
**identical** eligibility for identical grounded facts; they may take different trajectories,
but they cannot disagree about truth. A model cannot remove a candidate, and a model that
declares a violating candidate acceptable changes nothing, because the projection is built from
the deterministic verdicts.

### Task scope, and no stale verdicts

Eligibility is a statement about the current constraint set, not a permanent property of a
product: the same product is `INELIGIBLE` under `price_max=120` and `VERIFIED_ELIGIBLE` under
`price_max=200`. The evaluator holds **no cache at all** - every view is recomputed from the
current active constraints and the current grounded facts - which is the simplest design that
cannot produce stale eligibility. The cost is one fact lookup per candidate per constraint over
the run's own grounded set: never a catalogue scan.

Only `TaskState.hard_constraints()` is consulted, so the pre-existing hardness rule decides
feasibility. An explicit constraint from this turn or this session qualifies; an inferred or
memory-derived signal does not, and soft preferences keep ordering candidates **inside** the
feasible domain rather than shrinking it.

A stated constraint the catalogue carries no attribute for is mapped to
`ConstraintKind.UNVERIFIABLE` rather than dropped. Dropping it would present a candidate as
compliant with a constraint that was never tested; keeping it makes every candidate
`UNRESOLVED`, which is the honest answer.

### What the policy is shown

`PolicyContext` gains `verified_eligible_count`, `ineligible_count`, `unresolved_count`,
`feasible_parent_asins` and `active_constraints`. These are counts, constraint `kind<=value`
labels, and a **narrowing** of the allowlist the policy already had - never the catalogue facts
behind a verdict, and never an identity that was not already grounded. Excluded identities stay
out of the prompt; the policy is told *that* something was excluded and *which constraint*
excluded it.

That bounded summary is what makes `UNKNOWN` useful rather than merely honest: a policy can see
that a candidate is unresolved and spend a step reading the missing fact, which is a legal next
action it could not previously identify. *Which* candidate that step may name is enforced
separately — see the reference boundary in section 9.

### Completion

`CompletionGuard` gained one input pair and one refusal code. It does **not** evaluate
constraints - the eligibility layer owns that - it only refuses to certify a claim of compliance
nothing supports:

* at least one candidate proved compliant -> end normally;
* nothing proved compliant but something still undecided -> `no_verified_compliant_candidate`,
  retryable, so the policy can gather the missing evidence;
* nothing proved compliant and everything disproved -> an honest "no compliant match". The
  projection presents an empty set, so nothing violating is shown, and the guard does not refuse:
  spinning until the budget expired would settle nothing that is not already settled.

### Evaluation

The plane measures **checked** and **enforced** as separate questions over separate sets:

| Metric | Set | Question |
| --- | --- | --- |
| `constraint_checked` | everything the run evaluated | was a deterministic verdict obtained? |
| `constraint_violations` | what the response was built from | did a proved violation reach the answer? |
| `constraint_unresolved_in_output` | what the response was built from | was an undecided candidate presented as compliant? |
| `constraint_not_enforced` | what the response was built from | did a violating candidate reach the answer? |

They were one question over one set, which made "checked and compliant" indistinguishable from
"never checked" and made "excluded the violation" look identical to "presented it". A case's
declared `hard_constraints` now become the run's **real** active constraints, so a case that
asserts enforcement is measured against a run that actually enforced.

### Not implemented (documented, not claimed)

* **No candidate narrowing metric.** `constraint_not_enforced` detects a violation reaching the
  answer; it does not score *how well* a set was narrowed.
* **No ordering interaction with the reranker.** The accepted reranker still orders the same
  candidate set (section 15); it is simply handed the feasible one, so it cannot resurrect an
  ineligible candidate. No fused or learned ranker is introduced here.
* **Unmappable constraints are unresolved, not resolved.** A `feature` or `material` hard
  constraint has no deterministic catalogue dimension, so it makes every candidate `UNRESOLVED`
  rather than being decided. That is deliberate and is not worked around by inventing data.
* **The evaluator is not a service.** It is a pure function over the run's own grounded
  candidates; there is no incremental or index-backed eligibility structure, because a bounded
  per-turn set does not need one.

---

## 12. Evaluation planes (Stage 6, IMPLEMENTED)

Three planes, deliberately **not** collapsed into one score.

| Plane | Question | Where |
| --- | --- | --- |
| Recommender | where does the target item land in a full-catalogue ranking? | `recommendation/evaluation/metrics.py` (unchanged) |
| Shopping quality | does the final set respect stated preferences and constraints? | `recommendation/control/grounded_reasoning.py` verdicts |
| **Agent / trajectory** | did the run choose sensible actions, respect its budgets, recover, and terminate for a defensible reason? | `recommendation/evaluation/agent/` |

The separation matters because the metrics are not interchangeable. NDCG measures ranking
position and says nothing about whether an agent asked a useful question or recovered from an
empty retrieval; a trajectory metric says nothing about recommendation accuracy. Nothing in
`recommendation/evaluation/agent/` reuses an NDCG number, and no metric here claims relevance.

### Trajectory

Evaluation-plane version 4 adds `read_signature` to each record: for a read-only step, a bounded
signature of what it read (`get_details:cand-red,cand-blue`). It carries identities the record
already implies and no catalogue value, and it exists so a repeated read is distinguishable from
a read of different targets.

`AgentTrajectory` is the evaluation-facing projection of a run: per-step proposed action, whether
it was authorised, the refusal code if not, the observation kind/status, candidate and ungrounded
counts, whether the step consumed a tool call, the **policy that decided the step**, and the
terminal outcome. It is versioned and **payload-free** - no trusted history, no memory key, no
entry id - and a test asserts it.

Every step carries `policy_name`, taken from the runtime's own policy metadata. The recorded name
is identity only and never a decision input; a policy that reports no diagnostics still has its
declared `name` recorded by the loop. This exists because a variant *label* is a claim, and a
report that carried only the label could not distinguish a model-driven run from a silent fallback
to the default policy. That is not hypothetical: an ablation arm once passed
`policy_factory=None` and ran the deterministic stub while being filed as `model_policy`, which is
why the label is now corroborated by the run and asserted by a regression test.

### Cases

Twelve inspectable cases covering the ten required categories. Each declares more than a final
answer: required / acceptable / forbidden actions, expected sources, hard constraints, expected
memory effect, allowed terminal states and budgets. A run that reaches the right answer by way
of a forbidden action or an unperformed check fails.

### Metrics and attribution

`TrajectoryMetrics` reports each dimension separately and has **no ``score`` field**.
`FailureAttribution` maps every failure dimension onto the owning component
(`candidate_grounding`, `policy_selection`, `sequencing`, `completion`, `memory_scope`,
`constraint_verification`, ...), and a test asserts the mapping is total - so a new metric cannot
be added without deciding who owns it, and an unattributable failure is reported as a protocol
finding rather than dropped. `SuiteReport` reports `failures_by_component`, never one number.

Two measurements are kept separate on purpose: **checked** vs **enforced**. A run can obtain a
deterministic verdict for every candidate and still present a violating product.

### Ablation

Variants are **policy injection, not runtime flags**, so no unsafe switch enters production:
`adaptive` (the policy may react to its observation), `decide_once` (one decision, then wrap up),
and — Phase 1 — `model_policy` (a model-driven next-action policy over a deterministic
observation-reactive double). The decisive comparison is on `empty-source-recovery-required`, where
the history source returns nothing:

```text
adaptive      PASS  recommend_from_history -> search_catalog -> finish
decide_once   FAIL  recommend_from_history -> finish -> finish
model_policy  PASS  search_catalog -> finish
```

That is `Action -> Observation -> Policy` versus `Decide Once -> Fixed Workflow`, measured.

Run it with `.venv/bin/python -m experiments.agent_evaluation_smoke`.

### Not implemented (documented, not claimed)

* **No published benchmark numbers.** The cases are inspectable fixtures, not a leaderboard.
* **No clarification-quality metric.** Whether a question was *good* is not measured, only
  whether one was asked when the case required it.
* **No token or cost accounting.** Tool calls and steps are counted; token usage is not measured
  because no LLM is in the loop.
* **No candidate-narrowing *score*.** The loop now narrows a presented set (section 11), and
  the plane detects a violation reaching the answer, but it does not score how well a set was
  narrowed.
* **The model variant is driven by a double, not a model.** It measures the runtime seam and the
  authority boundary, not a model's competence (section 10).

---

## 13. Product metadata and candidate-scoped RAG

Two stages, deliberately split:

1. **`recommendation/catalog/`** turns the official Amazon Reviews 2023 product-metadata
   file into a deterministic `parent_asin`-keyed artifact and exposes a read-only lookup.
   It performs normalization and coverage accounting only — no candidate generation, no
   scoring, no ranking. A candidate with no metadata record is a normal outcome
   (`MissingMetadata`), never a reason to drop or substitute it.
2. **`recommendation/rag/`** (`ProductEnricher`) retrieves supporting evidence **inside the
   already-fixed candidate set**, using lexical BM25 over that candidate's own metadata
   fields.

The boundary is the point:

| RAG may | RAG may not |
| --- | --- |
| select metadata evidence fragments for a candidate | introduce a new candidate |
| attribute each fragment to a source field and provenance | replace SASRec candidate generation |
| support preference matching and grounded explanations | reorder candidates by itself |

Every fragment is verbatim, attributed and provenance-labelled. Missing metadata is
reported as unavailable rather than filled with generated text. Any reordering is a
*separate later stage* (section 16) with its own frozen policy.

---

## 14. Preference memory

`recommendation/memory/PreferenceMemoryService` owns explicit conversational preferences.
It is a different domain from interaction history and shares no field, method or table with
it — this package has no representation for a `parent_asin` history at all, so chat text
cannot become a behavioural event even in principle.

* **Scope.** Every record is keyed by an explicit `user_key`. There is no process-global
  store. The demo derives one `user_key` per session.
* **Provenance.** Each entry keeps its source turn id and the exact user span that produced
  it, so no preference can exist without evidence of what the user actually said.
* **Lifecycle is explicit.** `ADD` creates an independent constraint; an explicitly
  corrective statement (`instead`, `instead of X`, `make that`, `I meant`) produces
  `REPLACE` and supersedes same-kind, same-polarity entries; an explicit retraction
  produces `REMOVE` tombstones. Replacement is *never* inferred from the preference kind, so
  "I don't want red" followed by "I don't want blue" keeps both. Superseded and removed
  entries are retained for audit and excluded from use.
* **Extraction is injected** and called only with user-authored text — never with history,
  candidates, metadata, retrieval evidence or model output.
* **Persistence.** An in-memory store for tests and a versioned SQLite store
  (`SQLitePreferenceStore`) for runtime, using only the standard library.
* **No ranking.** Memory does not reorder candidates. It feeds evidence (section 8), which
  feeds ordering (section 9).

---

## 15. Preference evidence

`recommendation/preference_matching/PreferenceCandidateMatcher` evaluates each **ACTIVE**
preference against each candidate's **already-attached** metadata and returns one record per
pair with a three-state status:

* `MATCH` — the metadata positively satisfies the preference;
* `VIOLATION` — the metadata explicitly breaks it (a forbidden value is present, or a
  numeric bound is crossed);
* `UNKNOWN` — the available metadata cannot decide.

`UNKNOWN` is deliberately not a boolean false. A positive preference whose listed value
differs (`prefer red` against `Color = blue`) is `UNKNOWN`, because a metadata record is not
an exhaustive product specification. Positive preferences therefore never produce a
violation; only negative categorical constraints and numeric bounds can. This conservative
asymmetry is a correctness choice with a visible cost: evidence coverage under-reports the
metadata that is actually present.

Support is declared per preference kind against the real metadata schema, so a kind no field
can express is `UNKNOWN` for a *reason* (`unsupported_preference_kind`) rather than looking
like missing data.

The matcher reads only what the enricher attached; it holds no catalogue handle, retriever
or store, so it cannot reach a product outside the candidate universe, and it never adds,
drops or reorders anything.

---

## 16. Deterministic reranking

`recommendation/reranking/PreferenceReranker` applies one frozen lexicographic key:

```text
(violation_count ASC, match_count DESC, original_rank ASC, item_id ASC)
```

1. fewer supported violations wins;
2. at equal violations, more supported matches wins;
3. remaining ties keep the original SASRec order (minimal movement);
4. `item_id` is a defensive final key guaranteeing a total order.

Why lexicographic rather than a weighted score: raw SASRec scores are uncalibrated and
preference evidence has no validated numeric scale, so the policy prioritises explicit
constraint adherence **without** numerically combining uncalibrated signals.

Invariants (asserted by tests and re-checked offline by M10C):

* the same candidates come out that went in — never added, removed, replaced or filtered,
  even when a candidate violates a preference;
* the raw SASRec score is copied exactly — no normalisation, penalty or combination;
* the input evidence report is not mutated;
* reranked ranks are contiguous `1..N`;
* the reranker reads only the evidence report — no store, retriever, model, Tool or network.

`reranking/evaluation.py` (M10C) is the **offline** diagnostic layer: displacement, top-k
overlap, adherence-at-k, coverage, policy consistency, violation protection and movement
attribution. It calls the production reranker exactly as serving does, so it cannot drift
from it. It is **not** part of the serving path — no runtime node imports it, and AST guards
enforce that.

An accepted M10C finding worth recording: with valid input (**unique original ranks**),
`item_id` is never reached, so it does not determine production ordering. A separate,
purely cosmetic issue is that the M10B tail `reason` label can be imprecise for the last
candidate; ordering is unaffected, and the label is deliberately not surfaced to users.

---

## 17. Web / session layer

Three layers, each with a narrow job:

| Layer | Module | Owns |
| --- | --- | --- |
| HTTP adapter | `recommendation/api/demo_routes.py` | request validation, session lookup, turn-id allocation, graph invocation, serialization, HTTP error mapping |
| Session state | `recommendation/demo/sessions.py` | session registry, per-session locks, turn sequencing, capacity, reset |
| Composition | `recommendation/demo/runtime.py` | building heavy collaborators once, per-`(k, user_key)` compiled-graph cache |

The controller contains **no** scoring, masking, sorting, retrieval, matching, reranking or
explanation logic, and imports none of those implementations (AST-guarded). It cannot: the
graph does not expose candidate scoring, and the graph is reached only as a whole.

**Session model.** A session owns an opaque UUID4 `session_id`, a **session-derived**
preference-memory `user_key` (never derived from the demo profile, so two sessions on one
profile stay isolated), the application-owned trusted history copied from a `DemoProfile`,
a logical creation index, a turn counter and its own `threading.Lock`. Session ids are
capability tokens for a local demo, not an identity system.

**Turn ownership.** Turn ids are `"<session_id>:<sequence>"`, allocated by the server while
holding that session's lock, and never accepted from a client. A failed turn still consumes
its id, so retries cannot reuse one. There is no global lock: different sessions proceed
independently.

**Public serialization.** `recommendation/demo/serialization.py` is an explicit **whitelist**
adapter from graph state to the public response — no `dict(state)`, no `model_dump()` of
internal models, no splat. Candidate metadata and evidence are attached by
`(parent_asin, item_id)` **identity**, never by list position, so a reordered candidate can
never inherit its neighbour's facts.

**Frontend.** Plain HTML/CSS/JS served as static files: no Node toolchain, no framework, no
CDN. All dynamic text is written with `textContent` / `createElement` (`innerHTML` and
friends appear nowhere), and the client contains no scoring, sorting, filtering or
re-ranking — it renders the API's sequence and the API's rank fields.

---

## 18. State ownership

| State | Owner | Lifetime | Writable by |
| --- | --- | --- | --- |
| Trusted interaction history | application / session | process (rebuilt from accepted artifacts) | application only |
| Preference memory | `PreferenceMemoryService` + store | persistent (SQLite) | the user's own explicit statements |
| Browser transcript | the page | the page | the page |
| Tool result, enrichment, evidence, reranking | graph run (derived) | one run | the stage that produced it |
| Live session registry | `DemoSessionManager` | process only | the session layer |
| Accepted artifacts | repository | durable | explicit retraining, to a new run directory |

Two rules follow. First, **original order is never overwritten**: `tool_result` and
`enrichment` keep SASRec order and their `rank` values are the authoritative
`original_rank`; the reranking report is additive derived state. Second, **one coherent
memory snapshot per turn**: the snapshot is read once at turn start and used for the whole
turn, so a write during the turn cannot influence it.

---

## 19. Trust boundaries

```mermaid
flowchart TD
    CHAT["chat text"]
    H["Trusted interaction history<br/>application-owned, chronological"]
    P["Preference Memory<br/>explicit preferences, per user_key"]
    T["Browser transcript<br/>page display state"]
    S["SASRec / RecommendationTool"]
    R["preference evidence + reranking"]

    CHAT -. "blocked" .-> H
    CHAT --> P
    CHAT --> T
    H --> S
    P --> R
```

1. **Chat text → interaction history is blocked.** Structurally: the decision step is
   `build_decision_messages(user_message)`, a function of one string with no parameter that
   could carry history; `AgentDecision` forbids unknown fields; the history channel has no
   reducer and no node writes it; the Tool reads history from `RecommendationContext` built
   from application state. Telling the demo *"I bought B0BX5QFWQN."* creates no interaction
   event, and this is asserted at the HTTP level.
2. **Chat text → preference memory is bounded.** Only explicit preference statements can
   create entries, and every entry carries the source span and turn that produced it.
3. **Preference memory → interaction history does not exist.** The memory package has no
   representation for a `parent_asin` history; the two domains share no type.
4. **Retrieval cannot widen the candidate set.** RAG operates only inside the candidates
   SASRec produced.
5. **Ordering cannot invent candidates.** The reranker emits a permutation of its input;
   the renderer re-verifies that by identity and raises if the candidate set changed.
6. **The browser is untrusted.** Request schemas are `extra="forbid"` and contain no field
   for history, `parent_asin`s, a preference snapshot, a reranking report, a `user_key`, a
   session id or a turn id.

---

## 20. Failure behaviour

The system prefers explicit failure to plausible degradation.

| Stage | On failure |
| --- | --- |
| Tool / engine | raises a Tool domain error; enrichment, matching and reranking are never reached |
| Enrichment | propagates; no preference evidence is fabricated |
| Matcher | propagates; no partially reranked list is produced |
| Reranker | propagates; the graph never falls back to a raw-order answer presented as if the policy ran |
| Renderer | raises `AgentGraphError` if the reranking report changed the candidate count or introduced an unknown identity |
| Session layer | unknown/expired/reset session → explicit 404; capacity → explicit 503; the server never silently creates a session |
| HTTP mapping | every `5xx` detail is authored by the mapping layer, so exception text, paths and stack traces never reach a client |

`DIRECT` is unaffected by any failure in the preference stages, because it never reaches
them. Candidate exhaustion is a **normal** outcome (`returned_k == 0`) and is rendered as a
neutral empty state — never padded with a fallback product. There is no popularity or random
fallback anywhere.

---

## 21. Determinism and reproducibility

* **Seeds and protocol are recorded.** The accepted run manifest pins seed 2026, the model
  and optimizer configuration, the evaluation protocol version, the cohort definition, the
  processed-artifact digests and the raw-data digest.
* **Checkpoint selection is validation-only.** The manifest asserts
  `test_sealed.test_used_for_selection = false`; the test set is opened once, after
  selection.
* **Ranking is deterministic.** The tie rule is a total order, so results do not depend on
  sort stability or platform.
* **Reranking is deterministic** and its reasons are derived from the canonical key rather
  than from labels.
* **Offline reports exclude timing and machine metadata** from their deterministic payload,
  so repeated runs share a digest. The accepted M10C real-cohort digest is
  `c8fe88efbdc96188…`, reproduced byte-identically by later milestones.
* **The graph is a pure function of its inputs** for a fixed `(k, user_key)`: the same
  script over fresh sessions produces identical order, ranks, evidence counts and text
  (session ids, turn ids and timings excluded).

---

## 22. Dependency lifecycle / heavy-object reuse

Constructed **once per process** by `DemoRuntime`:

| Object | Cost | Note |
| --- | --- | --- |
| `SASRecInferenceEngine` | ~349 MB checkpoint | verified against the accepted digest at startup |
| `RecommendationTool` | thin | wraps that one engine |
| `MetadataIndex` | ~300 MB artifact | parsed once, not per candidate or per turn |
| `ProductEnricher` | thin | wraps that one index |
| `PreferenceMemoryService` + store | one SQLite database | session-scoped namespaces inside it |
| `PreferenceCandidateMatcher` | stateless | one instance |
| `PreferenceReranker` | stateless | one instance |
| `DemoSessionManager` | bounded registry | `max_sessions`, optional TTL |

**The one genuinely per-request value is `k`**, because the accepted trust boundary routes
it exclusively through `AgentDecision` and there is no other channel. The runtime therefore
caches one compiled `AgentGraph` per `(k, session user_key)`: compiling a graph is pure
Python over the *same* process-scoped collaborators, the cache is bounded, and entries are
released when a session is reset. No model, index or service is ever rebuilt per request or
per turn.

The graph is bound to a memory namespace at construction (accepted M9 contract), which is
why the cache key includes `user_key` rather than one shared graph being mutated. This is a
deliberate choice to keep the frozen Milestone 7–10 components unchanged.

Startup fails fast when a required accepted artifact is missing, so a misconfigured server
refuses to start rather than failing on the first browser request. When an engine is
injected (tests), the demo reuses it instead of loading a second checkpoint.

---

## 23. Real provider evaluation with record and replay (Phase 3, IMPLEMENTED)

Phases 1-2.1 built a bounded policy seam, a deterministic model double, constraint enforcement
and a run-scoped reference boundary.  What none of them had done was put a **real model** through
that seam, so every "model policy" number was really a statement about the runtime.

This phase runs the real thing and makes the run reproducible.

```
LLMAgentPolicy
      |  ModelRequest
      v
RecordingModelClient                    one seam, four modes
      |
      +-- canonical fingerprint (provider + model + prompt version + context + schema + correction)
      |
      +-- REPLAY            : look up, return the recorded response, never touch the network
      +-- LIVE              : call, return, write nothing
      +-- RECORD            : call, return, write
      +-- RECORD_IF_MISSING  : replay when recorded, otherwise call and write
      |
      v
OpenAICompatibleChatAdapter             provider profile: thinking, JSON mode, temperature
      |
      v
urllib_transport                        standard library, imported lazily
      |
      v
DeepSeek-compatible endpoint
```

Only the **policy** varies across the evaluation variants.  The runner, controller, tools,
catalogue, candidate ledger, constraint kernel, reference boundary, completion guard and case
definitions are the same objects in every run - see `experiments/_harness.py`, which exists
precisely so the commands cannot drift apart.

### Provider integration

DeepSeek is OpenAI-compatible, so it reuses the existing adapter rather than a new one.  Two
documented differences are handled by a `ProviderProfile` declaration rather than by branches in
the request path:

* **thinking mode is on by default** and emits a chain of thought in `reasoning_content`.  The
  agent asks for *one structured action*, so the adapter asks for thinking to be **disabled** and
  never reads the reasoning field.  The repository never requests chain-of-thought and never
  records it.
* **thinking mode ignores `temperature`**, so the adapter only sends sampling parameters the
  chosen mode honours.

Configuration is environment-only (`AGENTRECX_LLM_BASE_URL`, `_MODEL`, `_API_KEY`, `_TIMEOUT`,
`_PROFILE`, `_JSON_MODE`, `_THINKING`, `_INPUT_PRICE`, `_OUTPUT_PRICE`).  The credential is read
at call time, placed in the request header, and never logged, recorded, or included in an
exception - `ProviderSettings.describe()` reports `api_key_present` as a boolean so a report can
say a run was configured without carrying a value that could be printed.

### Recording format (version 1)

JSONL, one exchange per line, so a crashed run leaves every earlier exchange readable.  Each
entry holds the format version, the provider, model and prompt version, the fingerprint, the
request (system prompt, bounded context payload, action schema, correction), the response (text,
model id, token counts, request id, finish reason, latency, estimated cost) and an informational
timestamp.  It holds **no** credential, **no** authorization header and **no** provider
reasoning.

Recordings go to `/tmp/agentrecx-recordings/` by default - **outside version control**, because a
recording contains a prompt and a model's verbatim answer: useful as an experiment artifact, wrong
as an unreviewed repository fixture.  A deployment that wants in-tree fixtures points the path at
a reviewed directory.

### Fingerprinting

Replay keys on a SHA-256 of a canonical JSON body containing the provider, model, prompt version,
the system prompt's hash, the full context payload, the action schema and the correction.  Sorted
keys and fixed separators make it independent of dict insertion order; a test recomputes it in a
fresh interpreter to prove it does not depend on hash randomisation.

It is deliberately **not** keyed on the user request alone: a policy decision depends on the
observation, the candidate references, the offered menu, the constraint state and the retry
correction, so each of those changes the key.

A fingerprint may legitimately repeat within a run (an identical context makes an identical
request).  The store keeps **every** response for a key and a replay consumes them in order, so an
overwrite can never hide a collision and an exhausted key raises rather than reusing an earlier
answer.

### Modes, and the guarantee that matters

`REPLAY` never falls back to the network.  A silent fallback would make a "reproducible" run
quietly depend on a live endpoint, which is the property the mode exists to provide.  A missing
key raises `recording_missing`; a malformed file raises with its line number; a recording from
another format version is refused rather than guessed at.

### What is measured

Model identity and usage travel from the client to the trajectory through the policy's own
reporting: `execution_mode` (`live` / `record` / `replay` / `scripted`), `model_provider`,
`model_endpoint_id`, token counts, model latency and estimated cost.  Latency is measured with a
**monotonic** clock; the trajectory additionally records whole-turn wall latency, so model cost
and trajectory cost are distinguishable.  Unknowns stay `None`: a provider that reports no usage
is never counted as zero, and a partial usage is never padded into a total that looks measured.

Cost is estimated **only** from explicitly configured per-million prices.  Without them the cost
is reported as unknown - never fabricated, and never presented as provider billing truth.

Token and latency totals are summed over *every* call including replayed ones, because a replayed
call carries the measurement taken when it was recorded; the `live_calls` / `replayed_calls`
counts are what tell a reader which kind of measurement they are looking at.

### Evaluation variants

| Variant | Policy | What it measures |
| --- | --- | --- |
| `adaptive` / `decide_once` | deterministic stubs | the bounded loop itself |
| `scripted_model` | `LLMAgentPolicy` over `ObservationReactiveModel` | the runtime under model-shaped control, offline |
| `live_model` | `LLMAgentPolicy` over a real provider (or a replayed trace) | real model behaviour |

`scripted_model` is the name Phase 1 used to call `model_policy`; it was renamed when a real
provider arrived, because a label that could mean either invited exactly the confusion the
evaluation exists to avoid.

### Commands

```bash
# one bounded real call: configuration, transport, parsing, proposal, usage, latency
.venv/bin/python -m experiments.llm_policy_smoke --mode live

# the full case suite against the real provider, recording every exchange
.venv/bin/python -m experiments.live_agent_evaluation --mode record --confirm-live

# reproduce it offline: zero network, zero cost, same decisions
.venv/bin/python -m experiments.live_agent_evaluation --mode replay
```

Live execution takes **two** deliberate flags and no environment variable can start a paid run on
its own.  `pytest` requires no network, no credential and no provider.

### Measured baseline (Phase 3)

The baseline was run against DeepSeek (`deepseek-flash`) with the unchanged case suite, thinking
mode disabled and `temperature=0`, recording every exchange.  Two runs are preserved because a
**parse-protocol** revision sits between them - the prompt was never touched.

| Run | Prompt | Parse protocol | Passed | Model calls | Tokens (in/out/total) | Model latency |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | v1 | v1 | **2/14** | 44 | 41 892 / 1 935 / 43 827 | 39 348 ms (894 ms mean) |
| Revised | v1 | v2 | **3/14** | 62 | 77 818 / 3 195 / 81 013 | 56 679 ms (914 ms mean) |

Failure attribution, by owning component:

| Finding | Cases | Owner |
| --- | --- | --- |
| `"arguments": {}` rejected for an action that takes none | 7 of 12 baseline failures | `POLICY_PARSING` (fixed by parse protocol v2) |
| Over-asks for clarification instead of using the candidates it grounded | 9 of revised failures | `POLICY_SELECTION` |
| Repeated an action instead of recovering from a refused FINISH | 1 | `POLICY_SELECTION` |
| Re-read the same facts repeatedly until the step budget ran out | 2 | `POLICY_SELECTION` |

The first row is the interesting one, and it is a **harness** defect rather than a model defect:
DeepSeek sent `"arguments": {}` for `RECOMMEND_FROM_HISTORY` on 20 of 44 calls, and
`ActionProposal` rejects `{}` because it is not `None`.  `{}` and "no arguments" state the same
thing for an action that declares no argument fields, so the model seam now folds one onto the
other - narrowly: a *non-empty* payload for an argumentless action is still refused, an empty
payload for an action that *does* take arguments is still refused, and `ActionProposal` itself is
unchanged.  Correcting this accounted for 7 of 12 baseline failures and produced **no** regression.

The remaining failures are genuine policy behaviour, not trust violations.  The real model
systematically prefers asking a clarifying question over acting on the candidates it grounded -
including cases where it had already retrieved and inspected them - and it repeats actions rather
than recovering.  **No trust boundary was reached in any failing case**: no candidate identity was
manufactured, no constraint verdict was overridden, no memory was committed and no `FINISH`
bypassed the guard.

### Phase 3.1: calibration, and the measured Prompt v1 / v2 comparison

Two things were wrong after Phase 3, and they were different in kind.

**The measurement.** The live command never passed a catalogue reasoner to the runner, so
`_check_constraints` returned an empty tuple, every declared hard constraint was reported as
unchecked, and a completed run was marked `unverified_completion`. `hard-constraint` was
therefore failed for a reason that had nothing to do with the model. The runtime was unaffected -
the loop does its own constraint evaluation - only the report was wrong. Fixed by supplying the
reasoner, and pinned by a replay of the Phase-3 recording that flips that one case.

The **clarification-suspension** semantics the phase was opened for turned out to be already
correct: a run that suspends with `ASKED_CLARIFICATION` maps to that terminal and is accepted by
a case that declares it, and is a failure for a case that forbids asking. That is now pinned in
both directions, together with the neighbouring rules - a `COMPLETION_REFUSED`, a
`BUDGET_EXHAUSTED` and a `FAILED` abort are each distinct terminals and none of them counts as a
successful clarification.

**The contract.** Prompt v2 adds three generic rules and changes nothing else - same action
schema, same payload builder, same parse protocol:

* clarify only when a decision-relevant fact is missing *and* no offered action could obtain it,
  where grounded candidates and their facts count as information the run already has;
* do not ask merely because the request is brief;
* do not repeat a read-only action whose observation added nothing - change source, change
  target, ask only if the first rule holds, or finish on the evidence.

Measured over the same 14 cases, same model, same catalogue, same tools, same budgets, same
parser: **prompt v1 4/14 → prompt v2 9/14**, with clarifications 9 → 4, repeated read-only
actions 18 → 7, model calls 64 → 54 and model latency 59.4 s → 50.2 s. Six cases changed verdict,
all six improved, none regressed.

A second, narrower parse fix was needed along the way and is versioned as
`POLICY_PARSE_PROTOCOL_VERSION=3`: the v2 run spelled "no arguments" as `[]` where the v1 run
used `{}`, and only the object form was folded. Both are the JSON spelling of absent, neither is
a payload, and a *non-empty* literal for an argumentless action is still refused.

Trajectory records gained `read_signature` (evaluation-plane version 4) - `get_details:cand-red` -
so "did the run repeat a read it had already performed" is answerable from the record. Without it
two `get_details` steps on different candidates were indistinguishable from two on the same ones.

### Not implemented (documented, not claimed)

* **No prompt tuning against individual cases.** Prompt v2 was written from the *pattern* the
  baseline showed, before the v2 run, and no case was modified afterwards. Both prompt versions
  and both result sets are preserved, and a test asserts the prompt contains no case id, no case
  message and no expected action sequence.
* **No `NoProgressGuard`.** Repetition fell from 18 to 7 occurrences under the prompt change
  alone; a deterministic guard remains a documented option, not an implementation.
* **The parse-protocol revision is versioned, not silent.** `POLICY_PARSE_PROTOCOL_VERSION` is
  part of the recording fingerprint, so a trace recorded under one interpretation of an answer
  cannot be replayed as though it were made under another.
* **No repeated sampling.** One recorded run per case is the baseline; multiple runs per case are
  an extension, not a Phase-3 requirement.
* **No assertion about model competence.** The evaluation reports where a real model's behaviour
  differs from the deterministic policies; it does not claim a general quality level.
* **No provider-agnostic guarantee.** The transport is generic and the profile is declarative, but
  only DeepSeek was actually called.

---

## 24. Active evidence acquisition and real SimilarItem retrieval (Phase 4, IMPLEMENTED)

Phase 3.1 measured a capability ceiling rather than a reasoning one. Two facts made it visible:
some stated hard constraints map to `ConstraintKind.UNVERIFIABLE`, so `check_constraint` returned
`UNKNOWN` **by construction**; and repeated `GET_DETAILS` re-read the same facts, which cannot
produce a different answer. The agent had no way to obtain genuinely new grounded information.

### Two new trusted sources

```
FIND_SIMILAR(seed)                        ACQUIRE_EVIDENCE(candidate, attribute)
      |                                          |
      v                                          v
SimilarItemSource                          EvidenceSource
 lexcial TF-IDF + numpy cosine              reproducible local artifact
 over the trusted catalogue                 (not the catalogue projection)
      |                                          |
      v                                          v
CandidatePlane -> GroundingVerifier        EvidenceStore  (parent_asin, attribute) -> value
      |                                          |
      v                                          v
CandidateLedger (similar_item provenance)  GroundedReasoner.check_constraint
      |                                          |
      +------------------+-----------------------+
                         v
              CandidateEligibilityEvaluator        <- unchanged, still the only authority
```

**`SimilarItemSource`** is a **lexical item–item index**: TF-IDF vectors over catalogue text with
cosine similarity, in `numpy` alone. It is explicitly *not* learned embedding retrieval and *not*
ANN — there is no trained encoder, no vector database and no approximate index. A lexical index is
reproducible from the catalogue, needs no artifact, and its scores are explainable. Scores are
recorded with `score_kind="item_item_similarity"` and are never fused numerically with a SASRec
logit or a BM25 value; the ledger's rank-based fusion uses order alone.

`FIND_SIMILAR` now takes an **explicit seed** (`FindSimilarArguments.seed_parent_asin`) instead of
reusing the seedless `SelectSourceArguments`. Before, the action could not say *similar to what*,
so an implementation would have had to choose a seed itself — exactly the implicit target selection
Phase 2.1 removed. The seed is validated against the run's grounded allowlist **before** any
retrieval, so a model can select a seed but cannot introduce one.

**`LocalEvidenceArtifact`** is a trusted source over a reproducible local JSONL fixture that is
deliberately **not** the catalogue projection. That separation is the point: a fact in the artifact
is unknown to a run until an `ACQUIRE_EVIDENCE` action reads it, so the
`UNKNOWN → acquire → SATISFIED` trajectory proves the capability rather than re-reading facts the
run already had.

### The authority split, and one verdict authority

| Actor | May decide |
|---|---|
| the model | **which** held candidate needs evidence, **which** attribute, **which** grounded seed to widen from |
| `EvidenceSource` | **what fact it holds**, with provenance — never a verdict |
| `EvidenceStore` | **what is known**, and whether two trusted sources disagree |
| `GroundedReasoner.check_constraint` | **the verdict** — the one place `SATISFIED`/`VIOLATED`/`UNKNOWN` is produced |
| `CandidateEligibilityEvaluator` | **task eligibility**, from those verdicts, unchanged |

`EvidenceItem` has no verdict field at all, so a model-authored claim has no path into the store
and no path to a verdict. `EvidenceExecutor.execute` checks run membership before consulting any
source, exactly as the reasoning executor checks its targets.

### Three-state evidence semantics

Acquisition distinguishes `FOUND` / `NOT_FOUND` / `UNSUPPORTED`, and these are **not** verdicts:

* evidence states a satisfying value → `SATISFIED`;
* evidence states a non-satisfying value → `VIOLATED`;
* evidence states nothing, or the source cannot answer → `UNKNOWN`, **never** a violation inferred
  from silence.

`ConstraintKind` gained `MATERIAL` and `FEATURE`, decided from acquired evidence rather than from
the catalogue. `UNVERIFIABLE` is preserved for constraints nothing can decide. A stated material or
feature constraint therefore became resolvable instead of permanently unknown — but only through
acquisition, and a candidate the source does not cover stays `UNRESOLVED` and is never presented as
compliant.

If two trusted sources disagree, both provenance records are kept and the fact becomes **undecided**
(`UNKNOWN`), which is a conservative failure rather than a silent preference.

### Budget classification

`EVIDENCE_ACTIONS` is its own class: evidence acquisition is neither candidate-producing (the
candidate set is untouched) nor read-only reasoning (it reaches a separate trusted source for facts
the run does not have). It therefore consumes a tool call. `TOOL_CALL_ACTIONS = CANDIDATE_ACTIONS +
EVIDENCE_ACTIONS`; reasoning remains budget-free.

### Not implemented (documented, not claimed)

* **Not embedding or ANN retrieval.** The index is lexical TF-IDF cosine. A learned or approximate
  index would be a different implementation behind the same `CandidateSourceTool` protocol.
* **No external or web evidence.** The evidence source is a local reproducible artifact. Active
  external evidence retrieval is a later phase.
* **No `NoProgressGuard`.** Repetition is measured; a deterministic guard was deliberately not added
  (see the Phase-4 measurement).
* **No planner**, multi-agent split, RL/SFT, or commerce.

---

## 25. Public recommendation benchmark (Phase 5, IMPLEMENTED)

`experiments/benchmark_public.py` is the benchmark protocol and `experiments/phase5_benchmark.py` is
the runner. Nothing here is agent architecture: the phase exists to prove recommendation quality on
public data before the project is packaged as a finished system.

**One evaluator, one split, one candidate universe.** Every arm is scored by the accepted
`recommendation.evaluation.batched.evaluate_batched` over the same split
(`temporal_leave_two_out`, `agentrecx.eval_protocol.v1`), the same catalogue (`156 746` items), the
same `k ∈ {5, 10, 20}` and the same cohort. The evaluator owns PAD exclusion, seen-item masking,
tie-breaking and ranking; an arm supplies **raw scores** and never masks. A model arm yields a
`[batch, num_items + 1]` score matrix; a ranker arm encodes its ranking as a strictly decreasing score
matrix, so both are masked and ranked by the same code. `DEFAULT_BATCH_SIZE = 256` bounds peak memory
at ~3.9 GB, because the evaluator holds a float32 score matrix *and* a same-shaped bool mask.

**Five arms**, all on a deterministic 20 000-user history-length-stratified sample of the 412 445
eligible users: `popularity` (frequency over cohort train histories only), `sequential` (the accepted
SASRec checkpoint, no retraining), `metadata_retrieval` (BM25 item-to-item over catalogue text,
rank-fused across recent history items), `fixed_fusion` (RRF of all three sources for every user) and
`agent_selected` (per-user source selection over the same fusion). The runner refuses to write metrics
if the leakage gate fails.

**Measured outcome** (Recall@10): popularity 0.005 25 → metadata 0.009 25 → agent-selected 0.011 80 →
sequential 0.013 50 → **fixed fusion 0.014 35**. The accepted full-cohort SASRec reference is
0.013 57, which the sample's sequential arm reproduces (0.013 50). The result that matters for future
scope is that **fusion beats every single source, and threshold-based adaptive selection loses to
fusing everything** — the sources are complementary, so consulting one per user discards what the
others retrieved. Full detail, including the leakage evidence and the sample-versus-cohort caveat:
`docs/PHASE5_HANDOFF.md`.

---

## 26. External model backends (Step 2.2 specification, NOT IMPLEMENTED)

Custom **GenRec v0** (`recommendation/semantic_id/`, commit `80f81b1`) is **frozen as a historical
baseline**. It is not extended: no content embeddings, no k-means++ initialisation, no
collision-deduplication digit, no architecture change. The next generative-retrieval work is a
**separate public-TIGER backend**, specified in [`TIGER_BACKEND.md`](TIGER_BACKEND.md).

The boundary exists so AgentRec-X keeps every authority it already owns:

| Owner | Owns |
| --- | --- |
| **AgentRec-X** | `parent_asin ↔ item_id`, `PAD = 0`, `temporal_leave_two_out`, `agentrecx.eval_protocol.v1`, train-history exposure, the 20 000-user cohort, the full catalogue, seen-item masking, tie-breaking, ranking, Recall/NDCG/HR, `GroundingVerifier`, `CandidateLedger`, the Agent runtime, the candidate plane, fusion, SASRec, Two-Tower |
| **TigerBackend** | `item_id ↔ backend_row`, item content encoding, RQ-VAE / Semantic IDs, TIGER, catalogue-constrained and certified retrieval, raw item scores |

Four layers, with `parent_asin` permitted only in the outer two:

```text
L0  AgentRec-X canonical identity      parent_asin <-> item_id
L1  materialisation (experiments/)     reads parent_asin for text + required_frontier only
L2  TigerBackendAdapter                recommendation/backends/ - item_id ONLY, stdlib + NumPy
L3  TigerBackend                       backends/tiger_public/ - item_id ONLY, own venv
```

Three properties are structural rather than conventional:

1. **No canonical identity reaches the backend or the adapter.** `parent_asin` is not a
   parameter, field or return value anywhere under `recommendation/backends/` or `backends/`;
   the adapter's score conversion is a *column alignment*, not an identity translation.
2. **No target is visible to the backend.** The evaluation handoff carries histories and an
   integer `required_frontier = K_max + |seen|` only — no `test_target`, no `validation_target`,
   no seen-item identities, and no `grade_only` switch that would imply a protected field exists.
3. **The adapter holds no ML import and no backend import.** The crossing is a subprocess and a
   filesystem contract (JSONL / `.npy` / `.npz` / `.json`), with the backend in its own virtual
   environment so the root `requirements.txt` and the accepted CUDA/PyTorch stack are untouched.

The shared evaluator (`recommendation.evaluation.batched`) is **unchanged**: the backend streams
batched raw scores in the evaluator's own `[batch, num_items + 1]` item-id-column convention,
and the evaluator continues to own PAD exclusion, seen-item masking, tie-breaking, ranking and
metrics. Retrieval is reported as `APPROXIMATE` (diagnostic only) or `CERTIFIED` (required before
any comparison with SASRec or Two-Tower), and an uncertifiable run is reported as approximate
rather than presented as the definitive figure.

Steps: **2.3** skeleton and handoff bridge (no ML) → **2.4** content embedding + RQ-VAE +
collision-free Semantic IDs → **2.5** TIGER generator → **2.6** certified retrieval and the
canonical benchmark → **2.7** Agent integration, only once 2.6 establishes value.

### Not implemented (documented, not claimed)

* **No backend code, artifact or measurement exists yet.** Step 2.2 produced the specification
  only.
* **No item-content encoder, RQ-VAE, Semantic ID, TIGER or retrieval module is implemented**
  under `backends/`.
* **No `CandidateSource` member, no Agent tool and no fusion change** is introduced, and none is
  planned before Step 2.7.
* **No replacement of the frozen protocol, cohort, evaluator or identity mapping.**

---

## 27. Known limitations

* **Preference extraction is conservative and rule-based**, behind an injected seam.
* **Evidence coverage can be sparse** by design: a readable field holding a different value
  is `UNKNOWN`, so coverage under-reports available metadata.
* **No preference-conditioned relevance labels and no user study** — policy adherence is
  measured, recommendation-quality improvement is not claimed.
* **No production authentication**; session ids are demo capability tokens.
* **The live session registry is process-local** and does not survive a restart; preference
  memory does.
* **Reset retains preference rows as unreachable audit provenance** — accepted memory
  semantics retain history and the web layer is forbidden from erasing store rows.
* **No learned Critic**; reranking is a frozen lexicographic policy.
* **Candidate generation is item-ID SASRec**; no Semantic IDs or generative recommender.
* **Candidate-scoped RAG is lexical BM25**, not dense retrieval.
* **The M10B tail reason label can be imprecise** for the last candidate; ordering is
  unaffected and the label is not user-facing.
* **ItemCF is not comparable** to the accepted full-data SASRec benchmark.
* **Hard-constraint enforcement is catalogue-bound.** A stated constraint with no
  deterministic catalogue dimension (`feature`, `material`, a free-form constraint) makes every
  candidate `unresolved`; it is never silently satisfied and never fabricated, so such a run
  presents nothing rather than guessing. Enriching the catalogue is the fix, not weakening the
  semantics.
* **Eligibility is recomputed per turn, not indexed.** It is a pure function over the run's own
  grounded candidates, which is correct and cheap at this scale but is not an incremental
  constraint index.
* **A reasoning refusal reports one code for both "not in this run" and "not in the catalogue".**
  The architecture distinguishes the two internally — membership is a refusal, a missing fact is
  `UNKNOWN` — but the primary check is membership, so an identity that is in neither fails on
  membership. Refining the message would tell a caller with no authority over a product whether
  that product exists.
* **The reference layer uses grounded identities, not opaque handles.** A handle table would add
  a resolution step without removing any exposure, because the identity is what the argument
  model is keyed by and every reference is re-validated against live run state anyway. If a
  future client needs identity-free prompts the projection is the single place to change.
* **Retrieval is lexical, not learned.** `FIND_SIMILAR` is TF-IDF cosine over catalogue text. It
  has no embedding model and no ANN index, so it cannot find a semantically similar product that
  shares no vocabulary.
* **Evidence is a local artifact, not live retrieval.** Phase 4 acquires facts from a reproducible
  fixture. There is no external or web evidence source, so a constraint the artifact does not
  cover stays `UNKNOWN` — which is the honest outcome, not a gap the model can reason around.
* **The public benchmark evaluates a 20 000-user sample**, not the full 412 445-user cohort, because
  this host is CPU-only with 7 GB of RAM. Comparisons between arms are controlled; absolute values are
  estimates. The sequential arm reproduces the accepted full-cohort figure (0.013 50 vs 0.013 57),
  which bounds the drift for that arm only.
* **The benchmark's agent-selected arm is a deterministic rule stand-in, not the live LLM policy.**
  A language model cannot be called once per user across 20 000 users. The rules are auditable, their
  firing rates are reported, and the negative result (selection loses to fixed fusion) is reported as
  measured rather than tuned away.
* **Benchmark absolute quality is low** (best arm Recall@10 = 0.014 35 over 156 746 items with one
  positive per user on a k-core-filtered category). It is a fair comparison, not evidence of a strong
  recommender.
* **The Phase-4 real-provider measurement exists, and its control is confounded.** Prompt v2
  unchanged, DeepSeek `deepseek-flash`, 18 cases recorded live (77 calls, 129 267 tokens, 81 324 ms
  model latency) and replayed exactly offline (77 replayed, 0 live, identical actions, terminals,
  failures, constraint verdicts and attribution).

  * **The capability works.** All three evidence cases passed against the real model, and it
    actually used both new actions: `acquire_evidence` 13 times and `find_similar` 3 times across
    the run. `evidence-satisfies` resolved `material` from `UNKNOWN` to a verdict and completed.
  * **Task success is 10/18.** On the 14 cases shared with the Phase-3.1 run it is 7/14 against
    the Phase-3.1 recording's 9/14 — but that comparison is **confounded** and should not be read
    as a capability regression: the Phase-3.1 figure comes from *replaying an archived run* while
    the Phase-4 figure is a *fresh live run*, so between-run model nondeterminism is mixed with
    the capability change. Additionally the Phase-4 harness offers `find_similar`/`select_source`
    for the Phase-4 cases, so the two arms do not see an identical action menu. A clean before/after
    would need both arms run fresh under the same menu.
  * **Retrieval is the model's weak point, not the source's.** `similar-item-recovery` failed
    because the model retrieved from history instead of the catalogue, then called `find_similar`
    three times on the same seed and asked a question — a policy-selection failure. Offline, the
    same case passes both policies and the index expands 1 candidate to 4.
  * **Every one of the 8 live failures is a policy-selection failure** — six over-clarifications
    and two no-progress read repetitions. None was an evidence-capability failure, a retrieval
    failure, or a runtime/trust-boundary failure.
* **Evidence acquisition cannot resolve a constraint the artifact does not cover.** Adding the
  capability does not add facts: a candidate outside the evidence artifact stays `UNRESOLVED`, and
  the run ends in an honest refusal exactly as it did before. The four new cases show both the
  resolution path and the not-found path deliberately.
* **The real-model experiment has a small sample.** One recorded run per case over the fixture
  catalogue (section 23): enough to measure where real behaviour differs in kind from the
  deterministic policies, not enough for a rate.
* **Single-provider bias.** Only DeepSeek was called. The transport and profile are generic, but
  no other provider was exercised, so nothing here supports a cross-provider claim.
* **Real models are not deterministic.** The baseline used thinking mode disabled and
  `temperature=0` for the most reproducible setting available, but identical live trajectories
  across calls are not claimed. Replay *is* deterministic, because it reproduces recorded text.
* **The cases exercise a 4-item synthetic catalogue, not real shopping.** The fixture is designed
  so that grounding, constraint and reference boundaries are observable; it does not model real
  catalogue scale, ambiguity or noise.
