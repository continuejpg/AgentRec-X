# Recommendation Tool (Milestone 7A)

An Agent-facing, framework-independent business layer above the accepted
Recommendation engine.

```
Future Agent
    ↓
RecommendationTool          (this package)
    ↓
SASRecInferenceEngine       recommendation/inference/
    ↓
SASRec model
```

## What it is

A thin contract that turns a **trusted** user interaction history into structured
recommendations. It owns exactly five things:

* the agent-facing input contract,
* trusted-context handling,
* business-level validation,
* calling the engine,
* normalizing engine output into a typed result, and mapping engine errors onto Tool
  domain errors.

## What it is not

* **Not a second recommender.** It does not load checkpoints, build mappings, encode
  histories, score, mask seen items, rank, or break ties — all of that belongs to
  `SASRecInferenceEngine` and is reused unchanged.
* **Not an HTTP client.** It calls the engine in process. It never routes through the
  FastAPI service. (Milestone 6's HTTP API remains an *external* boundary; the Agent is
  an internal component and talks to the engine directly.)
* **Not an agent.** No LangGraph, no LangChain, no LLM, no planner, no intent router,
  no tool-selection policy.
* **Not a generator of language.** It returns structured data only: no explanations,
  no summaries, no product descriptions.
* **Not a memory.** It accepts an already-resolved history. There is no `user_id`,
  session store, account store or profile store; mapping `user → history` belongs to
  later Memory/application-state work.

---

## Trusted-history boundary (mandatory)

> **User history must come from trusted application state.
> It must not be invented by an LLM.**

This is enforced by the shape of the API rather than by convention. The request type
and the history type are separate arguments of different types, and the request type
has **no history field at all**:

```python
tool.run(
    request=RecommendationToolRequest(k=10),        # what an agent/LLM may choose
    context=RecommendationContext(                  # trusted application state
        user_history=["B00...", "B07...", "B09..."],
    ),
)
```

`RecommendationToolRequest` accepts only `k` and forbids extra fields, so an LLM
choosing request arguments cannot add, replace or fabricate history. The Tool never
fetches history itself and never parses natural language.

---

## Request contract

`RecommendationToolRequest`:

| Field | Rules |
| --- | --- |
| `k` | integer, `1..100`, default `10`, **strict** (the strings `"10"` and the boolean `True` are rejected rather than coerced) |

There is intentionally no `budget`, `brand`, `color`, `category`, `query`, `intent`,
`constraints`, `location` or `natural_language_request`: SASRec cannot consume them,
and they belong to later Agent/RAG/Critic milestones.

## Context contract

`RecommendationContext`:

| Field | Rules |
| --- | --- |
| `user_history` | non-empty sequence of non-empty `parent_asin` strings, chronological |

* entries are stripped of surrounding whitespace; blank strings are rejected;
* duplicates are **preserved** — repeated interactions are real interactions;
* order is **preserved** — the Tool never reorders or sorts;
* the history is **never truncated** by the Tool: the engine owns model-window
  truncation, and the full history must stay available for seen-item masking;
* the caller's list is never mutated.

## Result contract

`RecommendationToolResult`:

```json
{
  "recommendations": [
    {"rank": 1, "parent_asin": "B0BX5QFWQN", "item_id": 151270, "score": 5.8085}
  ],
  "requested_k": 10,
  "returned_k": 10,
  "history_length": 8,
  "effective_history_length": 8,
  "history_truncated": false,
  "eligible_candidates": 156738,
  "timings_ms": {"scoring": 14.8, "ranking": 18.2}
}
```

Ranks, item ids, `parent_asin` values and scores are copied from the engine unchanged —
the Tool does not re-sort, filter, re-rank or transform them. No PAD and no already-seen
item appears.

### Score semantics

> The `score` is a raw **SASRec model score**. It is **not** a purchase probability,
> confidence, relevance probability, CTR or conversion likelihood. It is only
> meaningful for ordering candidates.

No product metadata (title, price, brand, description, image, category) is fabricated.

## Error behavior

Tool domain errors (`recommendation/tools/errors.py`) never carry filesystem paths,
checkpoint internals, stack traces or environment details:

| Situation | Error | Code |
| --- | --- | --- |
| invalid/missing request arguments | `InvalidRecommendationRequest` | `invalid_request` |
| no trusted history supplied | `MissingUserHistory` | `missing_user_history` |
| trusted history contains an unknown `parent_asin` | `UnknownHistoryItem` | `unknown_history_item` |
| engine unavailable / unexpected internal failure | `RecommendationUnavailable` | `recommendation_unavailable` |

All inherit from `RecommendationToolError`, so one `except` handles every Tool failure.
Each exposes `as_dict()` returning `{"error": code, "detail": message}`.

### Unknown-item behavior

Strict, matching Milestone 6: an unknown `parent_asin` fails clearly. It is **not**
dropped, **not** replaced with PAD, **not** re-mapped, and the call does **not** continue
with a partial history. A lenient mode would be a separate, explicit product decision.

### Empty-history behavior (known limitation)

SASRec is a sequential recommender and requires interaction history, so an empty
history fails with `MissingUserHistory`. There is **no** popularity fallback, **no**
automatic ItemCF call, and **no** random recommendations. Cold-start handling is out of
scope for Milestone 7A.

### Candidate exhaustion

Engine behavior is preserved and is **not** an error:

* fewer than `k` unseen items → `returned_k < requested_k`;
* no eligible items → `recommendations = []`, `returned_k = 0`.

---

## Engine ownership and lifecycle

The engine is injected once and reused for every call:

```python
engine = SASRecInferenceEngine(InferenceConfig(...))   # loaded once
tool = RecommendationTool(engine)                      # reuses it
```

The Tool never loads `best.pt`, the mapping or `run.json`, and performs no file I/O.
Repeated calls do not reload the model (asserted by test), and argument validation of
the engine is structural (`RecommendationEngine` protocol), so a test double, CLI,
background job or future LangGraph node can supply any conforming engine.

## Metadata

```python
tool.name         # "recommend_products"
tool.description  # "Generate SASRec recommendations from trusted chronological interaction history."
tool.version      # 1
tool.metadata()   # {"name": ..., "description": ..., "version": ...}
```

Framework-neutral: no LangGraph/LangChain decorators or base classes. The class has no
base classes beyond `object`.

## Usage

```python
from recommendation.inference import InferenceConfig, SASRecInferenceEngine
from recommendation.tools import (
    RecommendationContext, RecommendationTool, RecommendationToolRequest,
)

engine = SASRecInferenceEngine(InferenceConfig(
    checkpoint_path="runs/sasrec_canonical_2026/best.pt",
    mappings_path="data/processed/Sports_and_Outdoors_mappings.json",
    manifest_path="runs/sasrec_canonical_2026/run.json",
    device="cpu",
))
tool = RecommendationTool(engine)

result = tool.run(
    RecommendationToolRequest(k=5),
    RecommendationContext(user_history=["B00...", "B07...", "B09..."]),
)
for item in result.recommendations:
    print(item.rank, item.parent_asin, item.score)
```

## Smoke and latency

```bash
.venv/bin/python -m experiments.recommendation_tool_smoke
.venv/bin/python -m experiments.recommendation_tool_smoke --k 5 --calls 60 --json /tmp/tool.json
```

On CPU with the accepted checkpoint, the Tool wrapper adds roughly **0.02 ms p50** on top
of the ~32 ms engine path, i.e. overhead is negligible relative to inference.

## Tests

```bash
.venv/bin/python -m pytest tests/test_recommendation_tool.py tests/test_recommendation_tool_integration.py -q
```

`tests/test_recommendation_tool.py` uses a fake engine to prove pass-through behavior
without loading a model; `tests/test_recommendation_tool_integration.py` adds a
network-disabled proof (socket creation is monkeypatched to fail) plus a bounded real
accepted-checkpoint smoke. CPU only.

## Milestone 7B boundary

The LangGraph adapter this section anticipated now exists in
[`../agent/`](../agent/README.md), and it consumes the Tool exactly as documented
here:

```python
result = tool.run(request, context)   # history injected from application state
```

The core Tool stays framework-free, so it can still be driven by a CLI, a batch job
or a plain Python application. The agent layer adds orchestration, not recommender
semantics.

## Milestone 7C — real-chain integration

Milestone 7C proves this Tool composes with the accepted LangGraph graph and the real
engine + accepted Milestone 5 checkpoint. The Tool itself is **unchanged**: the
integration builds `engine → Tool → graph` once, injects a deterministic decision
model, and asserts that the request contract (`k` only) and the trusted-context
contract (application-owned history) hold on the real path, with no internal HTTP
hop through the Milestone 6 service.

```bash
.venv/bin/python -m pytest -q tests/test_agent_tool_e2e.py
.venv/bin/python -m experiments.agent_tool_e2e_smoke
```

Milestone 7C validates integration, not recommendation quality, and adds no product
metadata or semantic enrichment — those remain deferred to Milestone 8. See
[`../agent/README.md`](../agent/README.md) for the full path and boundary table.
