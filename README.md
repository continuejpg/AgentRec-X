# AgentRec-X

**Trustworthy Conversational Product Recommendation System**

A sequential recommender exposed through a conversational orchestration layer, where explicit
user preferences can change what gets shown — without giving the agent the authority to invent
product identities or alter the recommender's behavioural knowledge.

```text
Sequential Recommendation  ·  Conversational Preferences  ·  Persistent Memory
Grounded Evidence          ·  Deterministic Reranking      ·  Reproducible Evaluation
```

This is an offline research/demo system, not a production commerce stack. The serving path uses
no hosted LLM and needs no API key; every number below comes from a committed artifact or a
verified run.

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

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/demo/health` | readiness: `status`, `model_loaded`, `metadata_loaded`, `demo_ready` |
| `GET /docs` | interactive API docs |
| `GET /demo/` | browser demo (plain HTML/CSS/JS, no build step, same-origin assets only) |

Startup loads the accepted SASRec checkpoint and the ~300 MB catalogue metadata once (expect
roughly 20 s and an `Application startup complete.` line). Stop it with `Ctrl+C`. The demo runs in
the foreground; there is no PID file or daemon mode. **No hosted LLM API key is required — the
serving path is deterministic and offline.**

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

## Serving architecture

The browser demo path, by real component name:

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

One model generates candidates, and the browser path consults exactly that one model. Some of the
research extensions described further down are deliberately **not** on this path.

Canonical diagrams for both stacks — serving and offline research, with the trust boundary drawn
explicitly — are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

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

---

## Recommendation Trace

Each turn carries an optional, collapsible panel in the browser:

```text
Recommendation Trace
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
re-ranked.

It does **not** expose chain-of-thought, prompts, hidden reasoning or internal state. On the
browser path it also deliberately **omits** fields that have no authoritative value there — tool
name, retrieval source and catalogue-grounding counts are neither shown nor hardcoded. They exist
in the schema as optional fields so a future path can populate them honestly.

---

## Dataset and evaluation

| | |
| --- | --- |
| Dataset | Amazon Reviews 2023 — *Sports & Outdoors* |
| Interactions | 3,500,587 |
| Users / items | 412,445 users · 156,746 items (after iterative k-core filtering) |
| Split | chronological **temporal leave-two-out** — no random splitting |
| Frozen cohort | **20,000** users, deterministic seed `20260201` |
| Evaluation | full-catalogue ranking, seen-item masking, `Recall` / `HR` / `NDCG` @ K = 5/10/20 |
| Evaluator | one shared implementation (`recommendation/evaluation`) used by every model |

For single-positive leave-two-out, `HR@K` and `Recall@K` are numerically equivalent by
construction, and that is documented rather than treated as a bug.

---

## Frozen benchmark results

Same cohort, same protocol, same evaluator:

| System | Recall@10 | NDCG@10 |
| --- | --- | --- |
| Popularity | 0.00525 | 0.002687 |
| Metadata retrieval (BM25) | 0.00925 | 0.004405 |
| SASRec (serving path) | 0.01350 | 0.006848 |
| Two-Tower / DSSM | 0.01435 | 0.007365 |
| TIGER-FP32 (generative) | 0.01385 | 0.007755 |
| **Fixed rank fusion (comparator)** | **0.01435** | **0.007434** |

The standalone models sit within a few percent of one another on this cohort. These are small
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

## What happened when we evaluated a live LLM policy?

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
> work for recommendation, and it is **not** a test of adaptive source selection — that comparison
> was never run.

**Engineering lesson:** adding an LLM loop does not make a recommender adaptive by itself.
Action-space design, available tools, success criteria and candidate depth determine what a
bounded policy can actually do.

Full protocol and frozen report: [`docs/M3_PREREGISTRATION.md`](docs/M3_PREREGISTRATION.md),
[`docs/reports/m3-agent-vs-fixed-fusion.txt`](docs/reports/m3-agent-vs-fixed-fusion.txt).

---

## Repository organization

```text
recommendation/
  agent/                 original conversational orchestration (AgentGraph)
  models/  inference/    SASRec and supporting models
  tools/                 RecommendationTool contract
  catalog/  rag/         catalogue metadata index and candidate-scoped enrichment
  memory/                explicit preference memory (SQLite)
  preference_matching/   preference evidence (MATCH / VIOLATION / UNKNOWN)
  reranking/             deterministic preference reranking
  evaluation/            shared evaluator and metric primitives
  demo/  api/  web/      session layer, FastAPI app, browser UI
  control/               later bounded-control research track (offline)
experiments/             offline benchmarks and research tooling
docs/                    design, history, usage and research reports
runs/                    frozen experiment metadata and artifacts (git-ignored output)
backends/                external retrieval backend, own virtual environment
```

---

## Research track

The repository contains **two stacks that are not one identical runtime architecture**:

1. **The serving / demo stack** (above) — the original system: SASRec candidates, conversation,
   preference memory, evidence, deterministic reranking, browser demo. This is what the demo runs.
2. **A later research stack** — retrieval-model comparison (Two-Tower, Semantic-ID/TIGER),
   rank fusion, reproducibility work (hash manifest, frozen reports) and a preregistered
   evaluation of a bounded model-driven policy. It shares the dataset, protocol and evaluator but
   is not the browser serving path.

Deep-dive documentation, if you want the research detail:

| Topic | Document |
| --- | --- |
| Architecture and state ownership | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Milestone-by-milestone history | [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) |
| Experiments and results | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Canonical project state and status | [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) |
| Model expansion (Two-Tower, fusion) | [`docs/MODEL_EXPANSION_HANDOFF.md`](docs/MODEL_EXPANSION_HANDOFF.md) |
| TIGER backend boundary | [`docs/TIGER_BACKEND.md`](docs/TIGER_BACKEND.md) |
| LLM-policy preregistration | [`docs/M3_PREREGISTRATION.md`](docs/M3_PREREGISTRATION.md) |

---

## Limitations

- **The browser serving path is deterministic and offline.** There is no hosted LLM in it and no
  API key. A provider-backed policy exists only in the offline research track.
- **Preference extraction supports a bounded, explicit syntax.** It is a conservative rule-based
  parser, not general natural-language understanding. Anything outside that syntax is not stored.
- **Evidence matching is conservative and exact-token.** No synonym or stemming layer:
  `lightweight` is not supported by "ultralight", and `durable` is not supported by "durability".
  An unsupported positive preference reports `UNKNOWN`, because absent metadata proves nothing.
- **Not every retrieval model is in the serving path.** Two-Tower, TIGER and the fusion comparator
  are offline evaluation extensions.
- **Single-candidate-generator serving.** The browser agent uses SASRec; it does not perform
  adaptive multi-source routing.
- **Offline recommendation only.** No live inventory, pricing, cart, or CTR/CVR optimisation, and
  no online A/B-tested uplift claims.

---

## License and data

Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (*Sports & Outdoors*).
Raw data is never mutated in place; processed artifacts are reproducible from the raw input,
preprocessing configuration and code version. Large artifacts and model checkpoints are not
committed to Git.
