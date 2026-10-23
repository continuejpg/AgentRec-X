# AgentRec-X

**A bounded recommendation agent: multi-source retrieval, deterministic fusion, and an LLM that may propose but not decide what is true.**

AgentRec-X takes a conversational request and a user's behavioural history, retrieves candidates
with three independent sources, fuses their *orderings*, grounds each candidate in catalogue
metadata, and reranks by the preferences the user actually stated. Every step that determines what
a user sees — identity, scores, evidence, fusion, constraints, final ranking — is deterministic
code. The LLM, when enabled, proposes a route and preference values, and nothing else.

```text
Multi-Source Retrieval  ·  Deterministic Rank Fusion  ·  Catalogue-Grounded Evidence
Conversational Preferences  ·  Preference Reranking  ·  Full Recommendation Trace
Reproducible Offline Evaluation  ·  Preregistered Experiments  ·  Frozen, Hash-Recorded Reports
```

### Why this is not an "LLM wrapper around a recommender"

A typical LLM-plus-retriever demo lets the model choose products, quote facts, or rank results, and
then calls the output a recommendation. AgentRec-X inverts that: the model is a *proposal* layer
with no write access to anything that constitutes recommendation truth.

| | Typical LLM wrapper | AgentRec-X |
| --- | --- | --- |
| Candidate identity | model output | `parent_asin` ↔ `item_id` mappings; `PAD = 0` is never a candidate |
| Scores / ranking | model decides | model outputs, fused and reranked by deterministic code |
| Catalogue facts | model may paraphrase | `MetadataIndex`; read-only to the agent |
| Constraints | prompt instructions | typed preferences matched as `MATCH` / `VIOLATION` / `UNKNOWN` |
| Evaluation | anecdotes | one shared evaluator, frozen protocol, preregistered experiments |

The repository holds **two tracks that are not one runtime**, and the distinction is the first thing
worth understanding:

```text
PRODUCT — SERVING                            what /demo/ actually runs
  Conversation / Preference → Multi-Source Retrieval → Deterministic RRF
  → Catalogue Evidence → Preference Reranking → Result + Trace

OFFLINE LAB — RESEARCH ONLY                  never on the browser path
  models · fusion comparators · agent policies → shared evaluation → promote or reject
  Two-Tower study · TIGER / Semantic-ID · adaptive-routing policy · frozen reports
```

Everything on the serving path runs offline and needs no API key. Two-Tower began as an offline
study and is now a serving source because the evidence supported it; the offline comparison that
justified that promotion is reproduced in [Core offline evidence](#core-offline-evidence).

---

## 1. System architecture — PRODUCT (SERVING)

```text
User
  │
  ▼
FastAPI app  (recommendation/api/app.py) ── browser UI (recommendation/web)
  │
  ▼
AgentGraph  (recommendation/agent/graph.py) ── DemoSessionManager (per-session memory namespace)
  │
  ▼
RecommendationTool  (recommendation/tools) ── RecommendationEngine contract
  │
  ▼
Multi-Source Retrieval  (recommendation/inference)
  ├── SASRec            sequential, accepted checkpoint
  ├── Two-Tower         two_tower_cosine
  └── Catalogue Lexical whole-catalogue BM25 (bm25)
  │
  ▼
reciprocal_rank_fusion  (recommendation/control/candidate_ledger.py)
  RRF_K = 60 · source head = 100 · fused scores labelled rrf_rank_fusion
  │
  ▼
Catalogue Evidence  MetadataIndex + ProductEnricher  (recommendation/catalog, recommendation/rag)
  candidate-scoped: it annotates the fused set and can never widen it
  │
  ├───────────────────────────────┐
  ▼                               ▼
PreferenceMemoryService       PreferenceCandidateMatcher
(recommendation/memory)       (recommendation/preference_matching)
SQLite, per user_key          MATCH / VIOLATION / UNKNOWN
  │                               │
  └───────────────┬───────────────┘
                  ▼
      PreferenceReranker  (recommendation/reranking)
      deterministic lexicographic policy
                  │
                  ▼
      Recommendation Result  +  Recommendation Trace
      (recommendation/demo/serialization.py)
```

Design notes that matter for reading the rest of this document:

* **Only orderings are fused.** A SASRec logit, a Two-Tower cosine and a BM25 value are not
  comparable quantities, so the fusion consumes each source's *rank list* rather than summing
  incomparable scores.
* **Scores carry their meaning.** A fused score is labelled `score_kind = rrf_rank_fusion` and each
  recommendation carries provenance (which sources ranked it, at what rank), so a fused value can
  never be read as a model score.
* **The active engine is visible.** `/health` reports `active_engine`, `fallback_used` and
  `fallback_reason`. `AGENTRECX_SERVING_FUSION=sasrec` serves the sequential model alone; an `auto`
  deployment that cannot build the fused engine degrades to that path and says so over HTTP rather
  than only in a log.

Measured on a CPU-only development host (`python -m experiments.serving_fusion_latency`):

| | |
| --- | --- |
| Engine initialisation | ~44 s (two checkpoints + ~300 MB catalogue + the BM25 index) |
| Full demo startup | roughly 1–2 min on the tested CPU environment |
| Request latency | ~1.45 s median (k=10, history=20), of which the catalogue lexical source is ~1.2 s |

These are environment-specific CPU measurements, not serving SLAs: startup varies with disk speed
and the number of artifacts loaded, and request latency with host load. The lexical source is the
current performance bottleneck. It is recorded here rather than hidden.

---

## 2. Dataset and evaluation scale

| | |
| --- | --- |
| Dataset | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — *Sports & Outdoors* |
| Interactions | 3,500,587 (~3.5 M) |
| Users / items | **412,445 users** · **156,746 items** (after iterative k-core filtering) |
| Split | chronological **temporal leave-two-out** — never random |
| Frozen cohort | **20,000** users, deterministic seed `20260201` |
| Evaluation | **full-catalogue** ranking with seen-item masking, `Recall` / `HR` / `NDCG` @ K = 5 / 10 / 20 |
| Evaluator | one shared implementation (`recommendation/evaluation`) used by every model compared |

The dataset, the split and the evaluator are what the two tracks genuinely share: a shared evaluator
is what makes an offline comparison meaningful, and it does not by itself put a model on the serving
path. For single-positive leave-two-out, `HR@K` and `Recall@K` are numerically equivalent by
construction, which is documented rather than treated as a defect.

---

## 3. Core offline evidence

The clearest paired result in the repository asks one question: **does adding a Two-Tower retrieval
source to the existing candidate set help, on the same users, under the same frozen protocol?**

```text
A = Popularity + SASRec + Metadata
B = A + Two-Tower

Recall@10          0.01435  →  0.01900      (+32.4 % relative)
ΔNDCG@10           +0.0023818
95 % bootstrap CI  [+0.0017313, +0.0030716]   (user-level, 10,000 resamples, seed 20260201)
McNemar (hits)     b = 59, c = 152, p ≈ 1.2e-10
Verdict            SUPPORTED at K = 5, 10 and 20
```

A replication on a second arm pair (`C = sequential + metadata`, `D = C + Two-Tower`) moves in the
same direction: ΔNDCG@10 `+0.0021729`, CI `[+0.0011792, +0.0031958]`, McNemar `p ≈ 1.1e-06`.

### Evidence boundary — read this before quoting the number

```text
B1 paired evidence    : offline protocol, source depth 1000 (head lists read from a frozen
                        score matrix)
Serving V1 today      : source head 100, the deepest ranking the accepted serving engines
                        permit (MAX_K = 100)
```

The offline result establishes that fixed multi-source fusion has real complementarity on this
cohort, and it is what justified promoting Two-Tower into the serving path. It is **not** a
measurement of the current serving configuration:

* **the +32.4 % is an offline, full-catalogue, depth-1000 comparison**, not an online gain, not a
  depth-100 result, and not a claim about user satisfaction;
* no A/B test, click data or revenue measurement exists in this project, and none is implied;
* closing the depth gap would require a further experiment, which has not been run.

Offline model comparison on the same cohort, protocol and evaluator:

| System | Recall@10 | NDCG@10 |
| --- | --- | --- |
| Popularity | 0.00525 | 0.002687 |
| Metadata retrieval (BM25) | 0.00925 | 0.004405 |
| SASRec (serving path) | 0.01350 | 0.006848 |
| Two-Tower / DSSM | 0.01435 | 0.007365 |
| TIGER-FP32 (generative) | 0.01385 | 0.007755 |
| Fixed rank fusion (offline comparator) | 0.01435 | 0.007434 |

These standalone models sit within a few percent of one another on this cohort. TIGER was
**reproduced and integrated into the unified evaluation framework** as a generative retrieval
baseline; nothing in this table is presented as a win, and no claim is made that it outperforms
SASRec. An ItemCF baseline exists (`recommendation/baselines/itemcf.py`) but is deliberately **not
comparable** on this artifact — it is recorded as `itemcf_comparison = "PENDING (no same-artifact
full-data ItemCF benchmark exists)"`, so no ItemCF number appears above.

Two further offline studies are linked rather than summarised here, because both answered
"no promotion": a bounded **LLM agent policy** and an **adaptive source-selection policy** were each
preregistered, run once on the frozen 20,000-user cohort, and each lost to the fixed multi-source
comparator — see [`docs/M4_RESULT.md`](docs/M4_RESULT.md) and
[`docs/reports/m3-agent-vs-fixed-fusion.txt`](docs/reports/m3-agent-vs-fixed-fusion.txt). Their
engineering lesson is stated plainly there: adding an LLM loop does not make a recommender adaptive
by itself.

---

## 4. Quick start

```bash
./scripts/setup_demo.sh      # prepare if needed, then report readiness
./scripts/start_demo.sh      # start once the report says DEMO READINESS: READY
```

Then open **<http://127.0.0.1:8000/demo/>**.

`setup_demo.sh` is the single onboarding entry point. It creates `.venv` and installs dependencies
only when they are absent, then prints a readiness report ending in one of:

```text
DEMO READINESS: READY      → ./scripts/start_demo.sh can serve the product
DEMO READINESS: BLOCKED    → a required item is named, with the exact recovery command
```

* **No LLM key is required.** The default decision mode is deterministic and offline. Filling in
  `.env` (copied from `.env.example`) enables the optional LLM Agent decision mode; the launcher
  loads `.env` automatically, and an explicitly exported variable always wins over the file.
* **An artifact-recovery credential is optional.** It is needed only to download the runtime
  artifacts; a complete local set serves without one. A blocked report points at
  `scripts/fetch_artifacts.py verify` and `fetch --all`.
* **Startup takes about a minute** — two checkpoints and the ~300 MB catalogue are loaded once. The
  server runs in the foreground; `Ctrl+C` stops it. There is no daemon or PID file.

| Path | Command | Needs |
| --- | --- | --- |
| Readiness check | `./scripts/setup_demo.sh` | nothing beyond the checkout |
| **Serving V1** (the measured product) | `./scripts/start_demo.sh` | the runtime artifacts listed in `config/demo_runtime_artifacts.json` |
| Offline quick start | `./scripts/run_demo.sh` | nothing: it generates a small **synthetic** catalogue and a randomly initialised checkpoint |

Artifact recovery, verification tiers and port behaviour: [`docs/USAGE.md`](docs/USAGE.md),
[`docs/ARTIFACT_STORAGE.md`](docs/ARTIFACT_STORAGE.md).

| Endpoint | Purpose |
| --- | --- |
| `GET /demo/` | browser demo (plain HTML/CSS/JS, no build step) |
| `GET /health` | readiness plus the engine actually serving (`active_engine`, `fallback_used`) |
| `GET /v1/model` | non-sensitive model metadata: sources, `rrf_k`, source head, score kind |
| `GET /v1/demo/decision-modes` | which decision modes this deployment can serve, and the real reason when it cannot |
| `POST /v1/demo/sessions` | open a session (server-owned preference-memory namespace) |
| `POST /v1/demo/sessions/{session_id}/chat` | one conversational turn |
| `GET /docs` | interactive API documentation |

### What a session looks like

| # | You say | What happens |
| --- | --- | --- |
| 1 | `Recommend some useful gear for a weekend camping trip.` | 5 candidates from the fused retrieval; no preferences yet, so nothing moves |
| 2 | `I prefer lightweight and durable products.` | stored as **two** independent preferences: `feature: lightweight`, `feature: durable` |
| 3 | `Recommend the camping gear again using my preferences.` | reranked; `durable` reaches `MATCH` where the metadata supports it, `lightweight` stays `UNKNOWN` where the text only says "ultralight" |
| 4 | `Remove my lightweight preference.` | `route=direct`; the removal persists and is acknowledged from the stored result |
| 5 | `Recommend the camping gear again.` | only `durable` is active; ranking and evidence reflect that |

Preferences are stored independently rather than as one compound phrase; matching is exact-token and
conservative, with no synonym or stemming layer; reranking changes presentation order only, and the
original fused rank is shown next to the final one.

---

## 5. What the LLM can and cannot do

**Decision authority is narrower than truth authority.** This is the project's strongest design
constraint, and it holds whether or not a model provider is configured.

**The LLM may propose:**

* the turn route (recommend vs. direct answer);
* which stated preference values to store or withdraw.

**Trusted deterministic code owns — and the model cannot write to it:**

| Owned by trusted code | Where |
| --- | --- |
| the user's behavioural history | application-owned, read-only to the agent |
| canonical product identity | `parent_asin` ↔ `item_id` mappings; `PAD = 0` is never a candidate |
| catalogue facts | `MetadataIndex`; the agent may read, never author |
| retrieval and fusion | `reciprocal_rank_fusion`; fixed sources, fixed constant |
| model scores | the accepted checkpoints |
| preference evidence | `PreferenceCandidateMatcher` |
| reranking and constraints | `PreferenceReranker` |
| evaluation metrics | the shared evaluator |

In the optional LLM Agent decision mode the turn plan schema has fields for a route and for
preference values, and **no field at all** for a product id, a score, evidence, a rank or a mutation
result — with `extra="forbid"`, so an injected `parent_asin` is a validation error rather than an
ignored key. The plan is obtained *before* any trusted component runs, so a provider failure cannot
leave a partial preference mutation behind, and there is no silent fallback: a missing key, timeout
or malformed response returns an error code, changes no memory, and is never dressed up as a
deterministic answer.

Each turn carries a **Recommendation Trace**: decision mode, route, candidate count, active
preferences, memory changes, evidence counts and rank movement. It is a view of state the response
already contained — evidence is counted, never re-matched, never re-ranked — and it exposes no
prompt, chain-of-thought or hidden reasoning.

---

## 6. Repository organization

```text
recommendation/
  agent/                 conversational orchestration (AgentGraph)               [SERVING]
  models/  inference/    SASRec, Two-Tower, lexical source, fusion engine         [SERVING]
  tools/                 RecommendationTool contract                            [SERVING]
  catalog/  rag/         catalogue index and candidate-scoped enrichment          [SERVING]
  memory/                explicit preference memory (SQLite)                     [SERVING]
  preference_matching/   preference evidence (MATCH / VIOLATION / UNKNOWN)       [SERVING]
  reranking/             deterministic preference reranking                      [SERVING]
  demo/  api/  web/      session layer, FastAPI app, browser UI, readiness        [SERVING]
  evaluation/            shared evaluator and metric primitives                  [SHARED]
  control/               fixed fusion primitive; offline agent-policy evaluation  [SHARED/RESEARCH]
  baselines/             ItemCF engineering baseline                             [RESEARCH ONLY]
  semantic_id/           frozen historical GenRec v0                             [RESEARCH ONLY]
  backends/              adapter to the external TIGER backend                   [RESEARCH ONLY]
experiments/             offline benchmarks, arms, smokes and harnesses          [RESEARCH ONLY]
scripts/                 setup / start / artifact recovery
docs/                    design, usage, history and frozen research reports
runs/                    frozen experiment metadata and artifacts (git-ignored output)
```

Every package above has its own README with the contracts it enforces.

---

## 7. Project details and documentation

| Topic | Document |
| --- | --- |
| Architecture, both tracks | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| The same shape as pure text | [`docs/ARCHITECTURE_TEXT_DIAGRAM.md`](docs/ARCHITECTURE_TEXT_DIAGRAM.md) |
| Running it, artifact recovery, verification | [`docs/USAGE.md`](docs/USAGE.md) |
| Where every artifact lives and how it is verified | [`docs/ARTIFACT_STORAGE.md`](docs/ARTIFACT_STORAGE.md) |
| Experiments and frozen results | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Canonical project state | [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) |
| **Start here (new session / new contributor)** | [`docs/ONBOARDING.md`](docs/ONBOARDING.md) |
| **How the project was built and what it taught** | [`docs/ENGINEERING_PRACTICE.md`](docs/ENGINEERING_PRACTICE.md) |
| Every milestone brief, in order (regenerable) | [`docs/SESSION_MILESTONES.md`](docs/SESSION_MILESTONES.md) |
| Milestone-by-milestone history | [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) |
| Paired-evidence preregistration (B1) | [`docs/B1_PREREGISTRATION.md`](docs/B1_PREREGISTRATION.md) |
| Two-Tower design and fusion controls | [`docs/TWOTOWER.md`](docs/TWOTOWER.md) |
| Model expansion (Two-Tower, fusion) | [`docs/MODEL_EXPANSION_HANDOFF.md`](docs/MODEL_EXPANSION_HANDOFF.md) |
| Semantic-ID / GenRec baseline | [`docs/SEMANTIC_ID.md`](docs/SEMANTIC_ID.md) |
| External generative-backend boundary | [`docs/TIGER_BACKEND.md`](docs/TIGER_BACKEND.md) |
| LLM-policy preregistration (M3) | [`docs/M3_PREREGISTRATION.md`](docs/M3_PREREGISTRATION.md) |
| Formal adaptive-policy result (M4) | [`docs/M4_RESULT.md`](docs/M4_RESULT.md) |
| Presenting the demo | [`docs/INTERVIEW_DEMO.md`](docs/INTERVIEW_DEMO.md) |

### Limitations

* **The default serving path is deterministic and offline.** No hosted LLM is needed and no API key
  is required. The optional LLM Agent decision mode is a proposal layer only.
* **Preference extraction uses a bounded, explicit syntax** — a conservative rule-based parser, not
  general natural-language understanding. Anything outside that syntax is not stored.
* **Evidence matching is exact-token and conservative.** No synonym or stemming layer:
  `lightweight` is not supported by "ultralight", `durable` is not supported by "durability". An
  unsupported positive preference reports `UNKNOWN`, because absent metadata proves nothing.
* **The catalogue lexical source dominates request latency** (≈ 1.2 s of ≈ 1.45 s). Recorded, not
  optimised.
* **Serving fusion reads depth-100 source rankings**, while the offline paired evidence used depth
  1000 — see [Evidence boundary](#evidence-boundary--read-this-before-quoting-the-number).
* **Automated tests run on synthetic fixtures.** The suite never needs the real catalogue, a trained
  checkpoint or a provider, so it runs offline; only the live demo loads the accepted artifacts.
* **Offline recommendation only.** No live inventory, pricing, cart, or CTR/CVR optimisation, and no
  online A/B-tested uplift claims.
* **No relevance claim is made for explicit preferences.** The preference pipeline is evaluated for
  *policy adherence* — that a stored preference is applied as configured — not for whether the user
  liked the result. This repository measures **no preference-conditioned relevance labels**, and no
  reported figure is a satisfaction or recommendation-quality claim. Preference-conditioned
  relevance labels would require user feedback this project does not collect.
* **ItemCF is not a comparison point** on this artifact (see [Core offline evidence](#3-core-offline-evidence)).

---

## License and data

Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (*Sports & Outdoors*). Raw data
is never mutated in place; processed artifacts are reproducible from the raw input, the
preprocessing configuration and the code version. Large artifacts and model checkpoints are not
committed to Git — see [`docs/ARTIFACT_STORAGE.md`](docs/ARTIFACT_STORAGE.md) for where they live and
how they are verified.
