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
