# AgentRec-X

**Trustworthy Conversational Product Recommendation System**

A sequential recommender exposed through a conversational orchestration layer, where explicit
user preferences can change what gets shown — without giving the agent the authority to invent
product identities or alter the recommender's behavioural knowledge.

```text
Sequential Recommendation  ·  Conversational Preferences  ·  Persistent Memory
Grounded Evidence          ·  Deterministic Reranking      ·  Reproducible Evaluation
```

This is an offline demo and research system, not a production commerce stack. It contains **two
tracks that are not one runtime**, and telling them apart is the first thing to understand:

```text
PRODUCT — SERVING                        the path /demo/ actually runs today
  User → Conversation / Preference → Candidate Retrieval → Evidence
       → Preference Reranking → Result + Recommendation Trace
  Candidate Retrieval = Serving V1: SASRec + Two-Tower + catalogue lexical, combined by
  deterministic reciprocal-rank fusion (RRF_K = 60).  With AGENTRECX_SERVING_FUSION=sasrec,
  or when the Two-Tower artifacts are absent, it is SASRec alone — reported, never silent.

OFFLINE LAB — RESEARCH ONLY              not on the browser path (one opt-in exception below)
  Models / Fusion / Agent Policies → Shared Evaluation → promote or reject
  Two-Tower · TIGER / Semantic-ID · multi-source fusion · LLM adaptive routing
  · M3 / M4 · shared evaluator · preregistration, frozen artifacts, reports
```

The product path needs no hosted LLM and no API key; every number below comes from a committed
artifact or a verified run.

### In 30 seconds

| Question | Answer |
| --- | --- |
| **How does the current product recommend?** | **Serving V1**: `SASRec`, `Two-Tower` and a catalogue lexical retriever each propose an ordering, deterministic rank fusion (RRF, K=60) combines them, catalogue metadata supplies evidence, and a deterministic policy reranks. `AGENTRECX_SERVING_FUSION=sasrec` — or missing Two-Tower artifacts — serves SASRec alone, with the reason logged. |
| **Where does the LLM intervene?** | Only as a **proposal**: the turn route and preference values, and only in the opt-in LLM Agent decision mode. Its plan schema has no field for a product id, a score, evidence, a rank or a metric. |
| **Which models are offline research only?** | **TIGER / Semantic-ID, the popularity retriever, the research fusion comparator arms, and the LLM adaptive-routing policy.** Two-Tower is no longer in this list: it is a Serving V1 source since P2-A. |
| **What are M3 / M4, then?** | **Offline evaluation evidence** about candidate agent policies, each run once under a frozen protocol. They are not stages of the product: in both, the model-driven policy lost to the fixed multi-source comparator, so neither was promoted into the default serving path. |

### Three terms, used consistently below

| Term | Means | Does **not** mean |
| --- | --- | --- |
| **BM25 / catalogue evidence** | lexical grounding of a candidate against its own catalogue metadata | a production semantic retriever |
| **LLM** | a preference / intent **proposal** layer — and in the research stack, an action proposal | recommendation truth: it cannot supply identity, scores, evidence, ranks or metrics |
| **M3 / M4 control plane** | the **offline agent-policy evaluation** stack (`recommendation/control/`) | a serving architecture, or a product feature |

Two precision notes, because both are easy to get wrong:

- the bounded control-plane loop *can* be switched on in the demo as an opt-in alpha
  (`AGENTRECX_CONTROL_PLANE=loop`, off by default). Even then the demo registers **one** candidate
  source, because the multi-source candidate plane is constructed only by the offline harnesses;
- two retrieval models were evaluated and **not** wired in. That is a fact about the current
  implementation, not a claim that evidence rejected them — see
  [How the two tracks relate](#how-the-two-tracks-relate).

---

## What it does

- **Trusted behavioural history feeds a sequential recommender.** SASRec, trained on the full
  Amazon Reviews 2023 *Sports & Outdoors* category, produces the candidate set.
- **Conversational turns can store explicit preferences.** "I prefer lightweight and durable
  products" persists `lightweight` and `durable` as separate, typed preferences in per-user
  SQLite memory.
- **Catalogue metadata supplies evidence for those preferences.** Each candidate is matched
  against its own metadata (title, features, description, `details`) and reports `MATCH`,
  `VIOLATION` or `UNKNOWN` per preference.
- **A deterministic policy reranks the trusted candidates.** Fewer supported violations first,
  then more supported matches, then the original SASRec order. Original ranks are preserved and
  shown alongside the final ones.
- **The Recommendation Trace shows what happened.** Route, active preferences, memory changes,
  candidate count, evidence summary and rank movement — read from state the response already
  contained.

**The conversation can change presentation and ranking. It cannot rewrite the recommender's
knowledge, alter the model's scores, mask history, or invent a canonical product identity.**

---

## Run the live demo

```bash
./scripts/start_demo.sh
```

Then open **<http://127.0.0.1:8000/demo/>**.

Two paths, and the difference matters to a reviewer:

| Path | Command | Needs |
| --- | --- | --- |
| **Offline quick start** (recommended first) | `./scripts/run_demo.sh` | nothing: it generates a small synthetic catalogue and a **randomly initialised** checkpoint |
| **Full demo** (the measured configuration) | `./scripts/start_demo.sh` | the accepted SASRec checkpoint and the ~300 MB catalogue metadata |

**What a fresh clone can and cannot do.** The source, the protocols, the frozen reports and their
hash manifest are in-repo, and the offline path above is one command. The real checkpoints and the
catalogue are deliberately **not** distributed from this repository — eleven large artifacts, all
`durable_uri: NONE` ([`docs/ARTIFACT_STORAGE.md`](docs/ARTIFACT_STORAGE.md)) — so a fresh clone can
**inspect and verify** the recorded results but **cannot recompute every reported benchmark number
or serve the real-data demo** without those artifacts supplied out-of-band.

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/demo/health` | readiness: `status`, `model_loaded`, `metadata_loaded`, `demo_ready` |
| `GET /v1/demo/decision-modes` | which decision modes this deployment can serve, and the real reason when it cannot |
| `GET /docs` | interactive API docs |
| `GET /demo/` | browser demo (plain HTML/CSS/JS, no build step, same-origin assets only) |
| `POST /v1/demo/sessions` | open a session (server-owned preference-memory namespace) |
| `POST /v1/demo/sessions/{session_id}/chat` | one conversational turn |
| `GET /v1/demo/sessions/{session_id}` | session state and active preferences |
| `DELETE /v1/demo/sessions/{session_id}` | reset the session |

Startup loads the accepted SASRec checkpoint and the ~300 MB catalogue metadata once (expect
roughly 45 s with Serving V1 — it loads two checkpoints plus the ~300 MB catalogue — and an
`Application startup complete.` line). Stop it with `Ctrl+C`. The demo runs in
the foreground; there is no PID file or daemon mode. **No hosted LLM API key is required: the
default decision mode is deterministic and offline.** An optional LLM Agent decision mode can be
enabled with a provider configuration — see
[Optional LLM Agent decision mode](#optional-llm-agent-decision-mode).

Full detail, including artifact verification tiers and port-override behaviour:
[`docs/USAGE.md`](docs/USAGE.md).

> No screenshot is committed. Run the demo locally at `/demo/`.
>
> Presenting it live? Use the prepared operator script:
> [`docs/INTERVIEW_DEMO.md`](docs/INTERVIEW_DEMO.md).

---

## What a session looks like

A verified five-turn scenario on the real runtime:

| # | You say | What happens |
| --- | --- | --- |
| 1 | `Recommend some useful gear for a weekend camping trip.` | 5 SASRec candidates; no preferences yet, so evidence is empty and nothing moves |
| 2 | `I prefer lightweight and durable products.` | Stored as **two** independent preferences: `feature: lightweight`, `feature: durable` |
| 3 | `Recommend the camping gear again using my preferences.` | Reranked; `durable` reaches `MATCH` on candidates whose metadata contains the token, `lightweight` stays `UNKNOWN` where the text only says "ultralight" |
| 4 | `Remove my lightweight preference.` | `route=direct`; persisted removal; acknowledgement `Got it. I removed your lightweight preference. Active preference: durable.` |
| 5 | `Recommend the camping gear again.` | Only `durable` is active; ranking and evidence reflect that |

Observable behaviour worth noting:

- preferences are stored **independently**, not as one compound phrase;
- matching is **exact-token and conservative** — an unsupported term is reported `UNKNOWN`, never
  guessed, and there is no synonym or stemming layer (`lightweight` ≠ `ultralight`);
- reranking **reorders presentation only**; original SASRec ranks are unchanged and shown;
- removals persist and are acknowledged from the stored result, not from the user's wording.

Example rank movement from the verified demo profile:

```text
Original → Final
#1 → #1
#3 → #2
#4 → #3
#2 → #4
#5 → #5
```

This is an example, not a guarantee: it depends on the profile's trusted history and on which
candidate metadata happens to satisfy the stored preferences.

---

## Product architecture — SERVING

**This section, and only this section, is what the browser runs.** The diagram is the demo path by
real component name:

```text
User
  │
  ▼
FastAPI app  (recommendation/api/app.py)  ──  browser UI (recommendation/web)
  │
  ▼
AgentGraph  (recommendation/agent/graph.py)  ──  DemoSessionManager (per-session memory namespace)
  │
  ▼
RecommendationTool  (recommendation/tools)  ──  SASRecInferenceEngine, accepted checkpoint
  │
  ▼
SASRec candidate generation  (recommendation/inference, recommendation/models)
  │
  ▼
Catalogue metadata  ──  MetadataIndex + ProductEnricher  (recommendation/catalog, recommendation/rag)
  │
  ├──────────────────────────────┐
  ▼                              ▼
PreferenceMemoryService      PreferenceCandidateMatcher
(recommendation/memory)      (recommendation/preference_matching)
SQLite, per user_key         MATCH / VIOLATION / UNKNOWN
  │                              │
  └──────────────┬───────────────┘
                 ▼
      PreferenceReranker  (recommendation/reranking)
      deterministic lexicographic policy
                 │
                 ▼
      Grounded recommendation response
                 │
                 ▼
      Recommendation Trace  (recommendation/demo/serialization.py)
```

**Serving V1** is the frozen product shape: a fixed multi-source retrieval (SASRec + Two-Tower +
catalogue lexical) combined by deterministic rank fusion, then catalogue evidence, the preference
matcher/reranker, and a complete trace. The full shape, the measured performance and the evidence
boundary are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the contracts are in
[recommendation/inference/README.md](recommendation/inference/README.md).

## Trust boundary

The strongest design constraint in the project. **Decision authority is narrower than truth
authority.**

**The conversational / orchestration layer may:**

- interpret conversational intent inside its supported contract;
- **persist explicit preferences** the user states in their own words;
- invoke the recommendation path;
- affect **presentation and reranking** through those supported preferences.

**Trusted code owns — and the agent cannot write to it:**

| Owned by trusted code | Where |
| --- | --- |
| the user's behavioural history | application-owned, read-only to the agent |
| canonical product identity | `parent_asin` ↔ `item_id` mappings; `PAD = 0` is never a candidate |
| catalogue facts | `MetadataIndex`; the agent may read, never author |
| recommendation model output | accepted SASRec checkpoint and its scores |
| preference evidence | `PreferenceCandidateMatcher` |
| deterministic reranking | `PreferenceReranker` |
| evaluation metrics | the shared evaluator |

The agent can ask for candidates and can constrain what is *shown*. It cannot add a product that
is not in the catalogue, change a model score, expose consumed history as a recommendation, or
report a metric.

When the optional LLM Agent decision mode is selected, that boundary is unchanged — the model is
given **decision authority only**. It proposes one structured turn plan whose schema has fields for
a route and for preference *values* and **no field at all** for a product id, a score, evidence, a
rank or a mutation result, with `extra="forbid"`, so an injected `parent_asin` is a validation
error rather than an ignored key. Retrieval, masking, scoring, matching, reranking, persistence and
metrics all still run in the same trusted code, and the plan travels through the same accepted
Milestone 9 service a rule-extracted preference does.

---

## Recommendation Trace

Each turn carries an optional, permanent panel in the browser:

```text
Recommendation Trace
├── Decision             (Deterministic / LLM Agent, provider, proposed action, preference actions)
├── Route                (RECOMMENDATION / DIRECT)
├── Candidates           (count returned)
├── Active preferences
├── Memory changes       (added / removed / replaced)
├── Preference evidence  (MATCH / VIOLATION / UNKNOWN, counted per candidate)
└── Reranking            (original rank → final rank)
```

It is a **view of state the response already contained**: candidate counts come from the response's
own audit block, ranks and titles from the recommendation cards, mutations from the persisted
memory write result, and evidence is only counted — never re-matched, never re-extracted, never
re-ranked. The decision line is read from how the turn actually executed, so the panel can never
claim a policy the backend did not run.

It does **not** expose chain-of-thought, prompts, hidden reasoning or internal state. On the
browser path it also deliberately **omits** fields that have no authoritative value there — tool
name, retrieval source and catalogue-grounding counts are neither shown nor hardcoded. They exist
in the schema as optional fields so a future path can populate them honestly.

---

## Optional LLM Agent decision mode

The browser offers two decision policies:

```text
Decision Mode
[ Deterministic ]  [ LLM Agent ]
```

**Deterministic** is the default and needs no credential. **LLM Agent** is available only when a
provider is configured (`AGENTRECX_LLM_BASE_URL`, `AGENTRECX_LLM_MODEL`, `AGENTRECX_LLM_API_KEY`,
optionally `AGENTRECX_LLM_PROFILE`); `GET /v1/demo/decision-modes` reports the real availability,
and the browser disables the option with the reason rather than failing on the first turn.

When it is selected, one provider call produces one validated plan per turn. The plan decides the
route and which preference values are stored or withdrawn, and nothing else. Two properties are
enforced rather than promised:

- **the plan is obtained before any trusted component runs**, so a provider failure cannot leave a
  partial preference mutation behind;
- **there is no silent fallback** — a missing key, a timeout, malformed output or an unsupported
  action returns an error code (`llm_unavailable`, `llm_timeout`, `llm_invalid_plan`,
  `llm_provider_error`), changes no memory, and leaves the session in LLM mode, so a failure is
  never dressed up as a deterministic answer.

This is a *productisation* feature of the browser demo. It is not the research question answered in
[OFFLINE LAB: what happened when we evaluated a live LLM policy?](#offline-lab-what-happened-when-we-evaluated-a-live-llm-policy),
which measured a different system (the bounded loop control plane's adaptive policy) against a
fixed-fusion baseline under a frozen protocol. No LLM is used to produce any number in the
[benchmark results](#offline-lab-frozen-benchmark-results).

---

## Dataset and evaluation protocol (shared by both tracks)

| | |
|---|---|
| Dataset | Amazon Reviews 2023 — *Sports & Outdoors* |
| Interactions | 3,500,587 |
| Users / items | 412,445 users · 156,746 items (after iterative k-core filtering) |
| Split | chronological **temporal leave-two-out** — no random splitting |
| Frozen cohort | **20,000** users, deterministic seed `20260201` |
| Evaluation | full-catalogue ranking, seen-item masking, `Recall` / `HR` / `NDCG` @ K = 5/10/20 |
| Evaluator | one shared implementation (`recommendation/evaluation`) used by every model |

The dataset, split and evaluator are the one thing the two tracks genuinely share. Sharing an
evaluator is what makes an offline result comparable; it does not put a model on the serving path.

For single-positive leave-two-out, `HR@K` and `Recall@K` are numerically equivalent by
construction, and that is documented rather than treated as a bug.

---

## OFFLINE LAB: frozen benchmark results

> **RESEARCH ONLY.** Nothing in this section executes in the browser demo. These are offline
> comparisons of candidate models under one frozen protocol.

Same cohort, same protocol, same evaluator:

| System | Recall@10 | NDCG@10 |
| --- | --- | --- |
| Popularity | 0.00525 | 0.002687 |
| Metadata retrieval (BM25) | 0.00925 | 0.004405 |
| SASRec (serving path) | 0.01350 | 0.006848 |
| Two-Tower / DSSM | 0.01435 | 0.007365 |
| TIGER-FP32 (generative) | 0.01385 | 0.007755 |
| **Fixed rank fusion (comparator)** | **0.01435** | **0.007434** |

The standalone models sit within a few percent of one another on this cohort. An ItemCF baseline exists
(`recommendation/baselines/itemcf.py`), but it is deliberately **not comparable** on this artifact:
it is recorded as `itemcf_comparison = "PENDING (no same-artifact full-data ItemCF benchmark
exists)"`, so no ItemCF number is shown beside the table above. These are small
differences on a frozen offline benchmark; they are **not** presented as wins, and the fusion
increment was previously characterised as *directional only*.

Two retrieval families were also evaluated as **offline research extensions**, not wired into the
browser demo:

- **Two-Tower / DSSM** dual-encoder retrieval — [`docs/TWOTOWER.md`](docs/TWOTOWER.md)
- **Semantic-ID / TIGER** generative retrieval (RQ-VAE tokenizer + autoregressive generator,
  branch-and-bound certified catalogue retrieval) — [`docs/SEMANTIC_ID.md`](docs/SEMANTIC_ID.md),
  [`docs/TIGER_BACKEND.md`](docs/TIGER_BACKEND.md)

`fixed_fusion` likewise is an offline comparator: it combines `popularity` + `sequential` +
`metadata` by reciprocal-rank fusion. **The browser agent does not dynamically select between
these sources.**

---

## OFFLINE LAB: what happened when we evaluated a live LLM policy?

> **RESEARCH ONLY.** This is evaluation evidence about a candidate agent policy. It is **not** a
> stage of the product, and the default serving path does not run it.

After the demo system was complete, the project asked a harder question: does a *model-driven*
policy, given a bounded tool budget, do better than the deterministic fixed-fusion comparator
under the same frozen protocol? A preregistered 20,000-user evaluation was run once.

| | Agent policy | Fixed fusion |
| --- | --- | --- |
| NDCG@10 | 0.004554 | 0.007434 |
| Recall@10 | 0.00665 | 0.01435 |
| ΔNDCG@10 | **−0.002879**, paired 95% CI **[−0.003645, −0.002124]** | — |
| hit@10 (exact McNemar) | 28 gained / 182 lost, p ≈ 7.5e−29 | — |

The interval lies entirely below zero.

**Behaviour explains the result.** 19,999 of 20,000 trajectories were
`recommend_from_history → finish`. Catalogue search was never invoked. Source selection was
structurally unavailable in that frozen configuration, and the policy returned four candidates
against the comparator's fused ranking. The measured gap is a candidate-depth and
action-space effect as much as a ranking-quality effect.

> **The existing bounded LLM policy underperformed the accepted fixed-fusion comparator under the
> frozen protocol.**
>
> This is a result about one system configuration. It is **not** evidence that LLM agents do not
> work for recommendation, and it is **not** a test of adaptive source selection. That comparison
> was run later, as the offline **M4** experiment, where the model-driven adaptive policy also lost
> to the fixed multi-source comparator — see [`docs/M4_RESULT.md`](docs/M4_RESULT.md).

**Engineering lesson:** adding an LLM loop does not make a recommender adaptive by itself.
Action-space design, available tools, success criteria and candidate depth determine what a
bounded policy can actually do.

Full protocol and frozen report: [`docs/M3_PREREGISTRATION.md`](docs/M3_PREREGISTRATION.md),
[`docs/reports/m3-agent-vs-fixed-fusion.txt`](docs/reports/m3-agent-vs-fixed-fusion.txt).

---

## OFFLINE LAB: research track

**RESEARCH ONLY.** The repository contains **two tracks that are not one identical runtime
architecture**:

1. **PRODUCT — SERVING** (above) — the original system: SASRec candidates, conversation,
   preference memory, evidence, deterministic reranking, browser demo. This is what `/demo/` runs.
2. **OFFLINE LAB — RESEARCH ONLY** — retrieval-model comparison (Two-Tower, Semantic-ID/TIGER),
   rank fusion, reproducibility work (hash manifest, frozen reports) and preregistered evaluations
   of bounded model-driven policies (M3, M4). It shares the dataset, protocol and evaluator, but it
   is not the browser serving path.

### How the two tracks relate

```text
a new model / source / policy
        │
        ▼
OFFLINE LAB: shared evaluator + frozen protocol + preregistration
        │
   evidence sufficient?
     ┌──┴──┐
    yes    no
     │      │
     ▼      ▼
promotion   stays research only
candidate
```

The gate is a process, and its current rulings are deliberately uneven:

| Component | Status today | Why it is not on the serving path |
| --- | --- | --- |
| `SASRec` | **SERVING** | the accepted checkpoint; the sequential source of Serving V1 |
| Two-Tower + catalogue lexical | **SERVING** (Serving V1) | the other two Serving V1 sources, fused by RRF_K = 60 at source head 100; `AGENTRECX_SERVING_FUSION=sasrec` serves SASRec alone, and an `auto` fallback is visible on `/health` (`active_engine`, `fallback_used`, `fallback_reason`) — never only in a log |
| Two-Tower / DSSM | offline comparator | never wired into serving; promotion would need a serving-candidate audit (latency, artifact distribution, frozen fusion semantics) |
| TIGER / Semantic-ID | offline source, **qualified but not enabled** | qualified as a candidate source in the offline stack and explicitly not wired into the runtime |
| multi-source rank fusion | offline comparator | additionally blocked by an unimplemented step: the accepted preference reranker runs over the history recommender's candidates only, so *reranking over a fused candidate set* does not exist yet |
| LLM adaptive source selection | offline evidence, **not promoted** | it lost to fixed multi-source traversal under the frozen protocol (M4) |
| bounded control-plane loop | opt-in alpha, off by default (`AGENTRECX_CONTROL_PLANE=loop`) | available for evaluation; it is a second orchestration stack, not the product's default path |

The fusion row is **not** "evidence rejected fusion". The offline fusion results are a different
question from what the product runs, and what is missing there is an implementation and an audit —
stated as such rather than dressed up as a decision.

Deep-dive documentation, if you want the research detail:

| Topic | Document |
| --- | --- |
| Architecture, both tracks | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| The same shape as pure text | [`docs/ARCHITECTURE_TEXT_DIAGRAM.md`](docs/ARCHITECTURE_TEXT_DIAGRAM.md) |
| Milestone-by-milestone history | [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) |
| Experiments and results | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Canonical project state and status | [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) |
| Model expansion (Two-Tower, fusion) | [`docs/MODEL_EXPANSION_HANDOFF.md`](docs/MODEL_EXPANSION_HANDOFF.md) |
| Two-Tower design and fusion controls | [`docs/TWOTOWER.md`](docs/TWOTOWER.md) |
| Semantic-ID / GenRec baseline | [`docs/SEMANTIC_ID.md`](docs/SEMANTIC_ID.md) |
| TIGER backend boundary | [`docs/TIGER_BACKEND.md`](docs/TIGER_BACKEND.md) |
| LLM-policy preregistration (M3) | [`docs/M3_PREREGISTRATION.md`](docs/M3_PREREGISTRATION.md) |
| Formal M4 result | [`docs/M4_RESULT.md`](docs/M4_RESULT.md) |

---

## Repository organization

```text
recommendation/
  agent/                 original conversational orchestration (AgentGraph)      [SERVING]
  models/  inference/    SASRec and supporting models                            [SERVING]
  tools/                 RecommendationTool contract                            [SERVING]
  catalog/  rag/         catalogue index and candidate-scoped enrichment          [SERVING]
  memory/                explicit preference memory (SQLite)                     [SERVING]
  preference_matching/   preference evidence (MATCH / VIOLATION / UNKNOWN)       [SERVING]
  reranking/             deterministic preference reranking                      [SERVING]
  demo/  api/  web/      session layer, FastAPI app, browser UI                  [SERVING]
  evaluation/            shared evaluator and metric primitives                  [SHARED]
  control/               bounded control plane: offline agent-policy evaluation   [RESEARCH ONLY]
  baselines/             ItemCF engineering baseline                             [RESEARCH ONLY]
  semantic_id/           frozen historical GenRec v0                             [RESEARCH ONLY]
  backends/              adapter to the external TIGER backend                   [RESEARCH ONLY]
experiments/             offline benchmarks, arms, smokes and harnesses          [RESEARCH ONLY]
docs/                    design, history, usage and frozen research reports
runs/                    frozen experiment metadata and artifacts (git-ignored output)
backends/                the external TIGER retrieval backend, its own venv      [RESEARCH ONLY]
```

---

## Limitations

- **The default browser serving path is deterministic and offline.** No hosted LLM is needed and
  no API key is required. An opt-in LLM Agent *decision* mode exists in the product, and it is a
  proposal layer only; the offline research track's bounded policy is a different system.
- **Preference extraction supports a bounded, explicit syntax.** It is a conservative rule-based
  parser, not general natural-language understanding. Anything outside that syntax is not stored.
- **Evidence matching is conservative and exact-token.** No synonym or stemming layer:
  `lightweight` is not supported by "ultralight", and `durable` is not supported by "durability".
  An unsupported positive preference reports `UNKNOWN`, because absent metadata proves nothing.
- **Not every retrieval model is in the serving path.** Two-Tower, TIGER and the fusion comparator
  are offline evaluation extensions.
- **Single-candidate-generator serving.** The browser agent uses SASRec; it does not perform
  adaptive multi-source routing.
- **Automated tests run on synthetic fixtures.** The suite never needs the real catalogue, the
  trained checkpoint or a provider, so it runs offline; only the live demo loads the accepted
  artifacts.
- **Offline recommendation only.** No live inventory, pricing, cart, or CTR/CVR optimisation, and
  no online A/B-tested uplift claims.
- **No relevance claim is made for explicit preferences.** The preference pipeline is evaluated
  for *policy adherence* — that a stored preference is applied as configured — not for whether the
  user liked the result. This repository measures no preference-conditioned relevance labels, and no
  reported figure is a satisfaction or recommendation-quality claim. Preference-conditioned
  relevance labels would require user feedback this project does not collect.

---

## License and data

Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (*Sports & Outdoors*).
Raw data is never mutated in place; processed artifacts are reproducible from the raw input,
preprocessing configuration and code version. Large artifacts and model checkpoints are not
committed to Git.
