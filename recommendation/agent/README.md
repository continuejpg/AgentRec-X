# Minimal LangGraph Agent (Milestone 7B)

A deliberately small LangGraph workflow that proves the orchestration contract
around the already accepted Milestone 7A Recommendation Tool.

```
user message + trusted history
        │
        ▼
   ┌─────────┐
   │ decide  │   decision model: decide(messages) -> AgentDecision
   └────┬────┘
        │
        ├── action == "direct_response" ──────────────► finalize ──► END
        │
        └── action == "recommend" ──► recommend ─────► finalize ──► END
                                      │
                              RecommendationTool          recommendation/tools/
                                      │
                              SASRecInferenceEngine       recommendation/inference/
                                      │
                                 SASRec model
```

## What Milestone 7B is for

One thing: showing that LangGraph can drive the accepted Tool while the
trusted-history boundary stays intact. The graph has three nodes, two routes, no
cycle, and at most one Tool call per run.

## What it is not

Not a shopping assistant. Milestone 7B contains **no**:

* planner, task decomposition, ReAct loop, reflection, retry or summarizer;
* intent classifier beyond the single two-way route choice;
* critic, constraint checker or reranker;
* RAG, vector database, embedding, product-metadata retrieval or product search;
* memory, profile store or user database;
* semantic IDs, RQ-VAE or SID transformer;
* conversation history or multi-turn state;
* multi-agent hand-off or autonomous tool selection;
* FastAPI change — the Milestone 6 HTTP service is untouched, and the graph
  **never** calls it;
* retraining, new benchmark, evaluator/preprocessing/ItemCF change, or any change
  to Milestone 5 checkpoint or Milestone 6 inference/ranking semantics.

---

## Trust boundary (mandatory)

> **The LLM/Agent is not allowed to provide interaction history.
> The graph/application owns trusted history.**

Three structural mechanisms enforce this — not convention:

1. **No history-shaped argument exists.** The decision step is
   `build_decision_messages(user_message)`, a function of one string. There is no
   parameter that could carry history.
2. **A decision cannot smuggle history.** `AgentDecision` is a frozen pydantic
   model with `extra="forbid"`. A payload containing `history`,
   `trusted_user_history` or any other unknown field is a hard validation error.
   So an LLM cannot say `history = [...]` and change the recommendation history.
3. **Nothing writes the history channel.** `trusted_user_history` has no reducer
   in the graph state; the `decide` node returns only `decision` and the Tool node
   reads the history without writing it. The history a run starts with is the
   history it ends with.

### What the decision model must never receive

Verified by tests:

| Not passed to the decision model |
| --- |
| `trusted_user_history` |
| internal item ids |
| encoded model history |
| SASRec tensors |
| mapping internals |
| checkpoint internals |
| candidate lists or scores from a previous step |

The decision model receives exactly two messages: a policy system prompt and the
raw user message.

### Error messages never echo history

Rejected decision payloads and rejected `AgentInput` values are summarised by
failing field/rule only (`include_input=False`), so a validation message cannot
copy a guessed history value into a log line.

---

## Decision contract

```python
class DecisionModel(Protocol):
    def decide(self, messages: Sequence[DecisionMessage]) -> AgentDecision: ...
```

Injected, so Milestone 7B needs **no provider SDK** (`openai`, `anthropic`,
`google-generativeai` are not installed and not imported), **no `OPENAI_API_KEY`**
and **no internet**. A later milestone can supply a real adapter without changing
graph code. The call is synchronous because the graph is driven by LangGraph's
synchronous `invoke`.

`AgentDecision` accepts exactly one of two shapes:

| `action` | Required | Forbidden |
| --- | --- | --- |
| `recommend` | optional `k`, integer `1..100`, **strict** | `direct_response` |
| `direct_response` | non-blank `direct_response` (whitespace stripped) | `k` |

Anything else — unknown action, `"3"` or `true` for `k`, out-of-range `k`, missing
or blank `direct_response`, non-JSON text, a list, extra fields — raises
`MalformedDecision`. There is deliberately **no default route**: a broken decision
model is loud rather than silently recommending.

## Graph state

| Channel | Written by | Meaning |
| --- | --- | --- |
| `user_message` | application | the only natural-language input |
| `trusted_user_history` | application (read-only thereafter) | trusted chronological `parent_asin` tuple |
| `decision` | `decide` | the validated `AgentDecision` |
| `tool_result` | `recommend` | the Tool's typed result |
| `final_response` | `finalize` | the run's text |
| `route` | `finalize` | `direct` or `recommend` |

`AgentInput` validates the entry point by reusing the Tool's own
`RecommendationContext` schema, so the agent and Tool boundaries cannot drift.
Order and duplicates are preserved; the caller's container is never mutated.

---

## Honesty rules for the response

The candidate-only route states candidate `parent_asin` values and their raw model
scores, and nothing more. Every recommendation response carries:

> Note: these are raw sequential-model ranking scores used only to order the
> candidates. They are not probabilities or confidence values, and they are not
> evidence about a product's attributes, quality or availability.

**Milestone 7B provides no product semantic enrichment.** Candidate IDs and scores
are **not** sufficient evidence for product-attribute claims: this milestone has no
titles, brands, prices, categories, descriptions, images or reviews, and no
retrieval over product metadata. A later RAG milestone is what would supply such
evidence.

Candidate exhaustion is normal and is never turned into an error or padded with
fabricated items: `returned_k < requested_k` when fewer unseen items remain, and
`returned_k == 0` when none remain.

## Error behaviour

| Situation | Raised |
| --- | --- |
| malformed/unusable decision, or missing user message | `MalformedDecision` |
| run has no trusted history | `MissingUserHistory` (Tool domain error) |
| unknown `parent_asin` in trusted history | `UnknownHistoryItem` |
| engine unavailable / unexpected failure | `RecommendationUnavailable` |
| graph constructed with an unusable collaborator | `AgentConfigurationError` |

Tool domain errors propagate unchanged, so there is one stable taxonomy across the
Tool and the graph. `MalformedDecision` and `AgentGraphError` are the only
graph-level additions.

## Usage

```python
from recommendation.agent import AgentDecision, AgentGraph
from recommendation.tools import RecommendationContext, RecommendationTool, RecommendationToolRequest

tool = RecommendationTool(engine)          # Milestone 7A Tool, engine injected once

class MyDecisionModel:
    def decide(self, messages):
        # messages carries ONLY the system prompt and the user message
        return AgentDecision(action="recommend", k=5)

graph = AgentGraph(MyDecisionModel(), tool)
state = graph.run("suggest a tent", ("B00...", "B07...", "B09..."))
print(state["route"], state["final_response"])
```

## Smoke

```bash
# offline, deterministic, no checkpoint and no catalog needed
.venv/bin/python -m experiments.agent_graph_smoke
.venv/bin/python -m experiments.agent_graph_smoke --json /tmp/agent_smoke.json

# prove a malformed decision is refused instead of defaulted
echo '{"action": "browse"}' > /tmp/bad.json
.venv/bin/python -m experiments.agent_graph_smoke --decision-json /tmp/bad.json
```

The smoke proves exactly four things: the direct route (no Tool call), the
recommendation route over an injected fake Tool, malformed-decision rejection, and
the trusted-history boundary plus deterministic candidate/rendered output. It
requires no GPU, checkpoint, run directory, dataset, network or provider API — the
smoke has no flag that could load one.

## Tests

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py tests/test_agent_state.py -q
```

`tests/test_agent_graph.py` and `tests/test_agent_state.py` are fully offline and
deterministic: a duck-typed engine double plus an injected scripted decision model.
Normal pytest never loads `best.pt`, the 156,746-item catalog, a GPU or an external
API. The real chain (real `best.pt` -> engine -> Tool -> LangGraph) belongs to
**Milestone 7C**; nothing in Milestone 7B exercises it. The graph itself stays able
to accept any conforming injected `RecommendationTool`, so 7C can supply a real one
without changing graph code.

## Dependency

Milestone 7B adds exactly one direct dependency:

```
langgraph>=1.0,<2.0
```

Its transitive tree brings `langchain-core`, `langgraph-checkpoint`,
`langgraph-prebuilt`, `langgraph-sdk`, `langsmith` and small serialization/HTTP
helpers. None of them is an LLM provider SDK and none is imported by this package
except `langgraph.graph`. No provider client, no API key parsing and no network
call exists anywhere in `recommendation/agent/`.

---

## Milestone 7C — real-chain integration

**M7C validates integration, not recommendation quality.**

Milestone 7C does not change the agent. It proves that this accepted graph executes
against the accepted real stack:

```
trusted application history
    → AgentGraph (M7B, unchanged)
    → decision = RECOMMEND(k)      (injected deterministic DecisionModel)
    → RecommendationTool           (M7A contract, unchanged)
    → SASRecInferenceEngine        (M6, unchanged)
    → accepted M5 best.pt
    → full-catalog ranking + seen-item masking
    → structured result into the graph's final state
    → honest final response
```

### Dependency construction

The real chain is built **once per runtime** — `engine -> Tool -> graph` — and
reused for every invocation. Model reload per node or per call does not happen, and
the tests assert it via object identity and the engine's `loaded_at` timestamp
rather than private PyTorch internals.

Artifact discovery reuses the repository's existing configuration surface:
`recommendation.api.app.ServiceSettings` supplies repository-relative defaults
(`runs/sasrec_canonical_2026/best.pt`, its `run.json`, and
`data/processed/Sports_and_Outdoors_mappings.json`), honours the existing
`AGENTRECX_*` overrides, and its `to_inference_config()` supplies the engine config
including the accepted-checkpoint digest check. Only that configuration object is
reused — **the FastAPI app is never started and no HTTP request is made.** The
shared harness lives in `tests/agent_tool_e2e_runtime.py`.

### Trusted-history flow

History is selected deterministically from the accepted processed sequences
artifact: walk user records in stored order (ascending `user_int_id`), take the
**first** record with `length > 3`, and supply
`parent_asins[:-2] + [parent_asins[-2]]` — the training prefix plus the validation
target, with the final leave-one-out test target excluded, matching the accepted
M7A convention. Selection never inspects scores or candidates, so it cannot be
tuned toward a preferred output. The selected history is reported as a
length + one-way digest (`user_int_id=1`, length 5, digest `498eaa23691ca459`), and
its order and duplicates reach the engine unchanged.

The M7B trust boundary is unchanged on the real path: the decision model receives
only the user message, and the graph's trusted history is what the real engine gets.

### No internal HTTP

`AgentGraph → RecommendationTool → SASRecInferenceEngine` is entirely in process.
The M7C tests block `socket`, `create_connection`, `getaddrinfo` and `gethostbyname`
while the real chain runs, so an accidental HTTP hop would fail the suite.

### Deterministic decision model

M7C is not an LLM-provider milestone. The route is chosen by an injected
`SwitchableDecisionModel` (conceptually `ScriptedDecisionModel(AgentDecision(
action=RECOMMEND, k=5))`); no provider SDK, external API call, credential or online
inference is involved. A real LLM adapter, if ever required, is a later milestone.

### Real checkpoint identity

The E2E path asserts the accepted checkpoint digest
`352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912` and the accepted
`run.json` digest `e3549049f955bb0540444c221523444b2a333e8df5eac6b97984342e40c6e2c7`
before and after running. The artifacts are read-only: no retraining, refitting,
checkpoint selection, split change or metric recomputation occurs.

### Running the gate

```bash
.venv/bin/python -m pytest -q tests/test_agent_tool_e2e.py     # 36 tests, real chain
.venv/bin/python -m experiments.agent_tool_e2e_smoke           # 40 gates, prints evidence
```

M7C loads the 348 MB checkpoint and the 156,746-item catalog, so the test builds its
runtime once per session; every other suite stays lightweight. Both commands skip or
fail cleanly when the git-ignored artifacts are absent.

### Candidate exhaustion

With 156,746 catalog items and a longest real history of ~700 items, genuine
exhaustion (`returned_k == 0`) is **not reachable** on the accepted artifacts. The
reachable real-data case — a history longer than the maximum `k = 100` — is covered
by the E2E suite, and the authoritative exhaustion assertions remain the lower-level
M7A/M7B regressions. No catalog corruption was engineered to force the condition.

### Boundary summary

| Milestone | Scope |
| --- | --- |
| M7B | Offline orchestration contract; fakes only; **no** checkpoint, engine or catalog |
| **M7C** | Real-chain integration: injected deterministic decision model + real Tool + real engine + real checkpoint |
| M8 | Product metadata + candidate-scoped evidence retrieval (realisation of the row above) |
| M9 | Preference memory: explicit conversational preferences, loaded before `decide`, persisted after `finalize` (see `../memory/README.md`). Does **not** rerank. |
| M10 | Out of scope here: preference-aware scoring, critique and reranking |

**Product metadata and semantic enrichment belong to M8, not M7C.** M7C responses
expose only candidate identity (`parent_asin`), rank and raw model score; the
response states explicitly that scores are not probabilities and not evidence about
a product. M7C makes no claim that the recommendations are good, well personalised,
relevant or better than any baseline — that requires benchmark or qualitative
evaluation evidence this milestone does not provide.
