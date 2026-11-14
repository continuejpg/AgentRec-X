# AgentRec-X Final Demo V2 — Architecture

```text
Browser (Next.js 16 / React 19, web/)
   │  chat text, SSE, product cards, Memory panel, replay panel
   ▼
FastAPI BFF (web_demo/, 127.0.0.1:8787)
   │  session isolation at the browser-demo level (session ids); no identity injection
   ├── MCP stdio client A: read-only demo profile
   ├── MCP stdio client B: write-enabled demo profile (explicit local demo only)
   │
   ▼
recommendation.mcp.server  (thin adapter; no second Agent)
   ├── ElectronicsCatalogue + Original BGE / BM25
   ├── existing ElectronicsPlannerExecutor / Rule Planner / Gate
   ├── M9 PreferenceMemoryService + HistoryStateReader
   ├── M9 remember_preference (demo-only explicit write adapter)
   └── P15 evidence / P16 comparison
```

## Display layers

```text
Main chat + product shelf
  user intent, real assistant narrative, clarified ASK, top-K cards, evidence/UNKNOWN
        │
Active Memory + Memory Timeline
  real M9 active/inactive entries; logical sequence; supersession/withdrawal
        │
Agent Inspector (tool timeline)
  real tool rows, counts, Gate reason, selected IDs, latency
        │
Rankers / Memory OFF-ON (Replay)
  recorded P100 GPU predictions; explicit Historical Replay badge
```

## Responsibility boundary

| Layer | Owns | Never does |
|---|---|---|
| Next.js UI | layout, cards, timeline rendering, state, user input | no fake products, no reranking, no memory reset, no model claims |
| BFF | HTTP/SSE, sessions, payload → view model, replay read | no `recommendation.*` import in BFF; no ranking |
| MCP server | existing tools plus two thin demo adapters | no new agent, no new store, no hidden fallback |
| Existing Agent core | planner/executor/Gate/scorer/memory | unchanged |

## Memory path

```text
user-authored text
→ M9 extractor / validation
→ PreferenceMemoryService.process_turn
→ permanent audit rows
→ HistoryStateReader active state
→ executor preferences
→ existing PreferenceAwareScorer
→ DecisionGate / TopK
```

Withdrawal does not delete history rows. The timeline renders the removed/inactive row and its
source turn; the active panel no longer displays it.

## Replay path

```text
runs/p100_extended_evaluation/p100_ranker_gpu_predictions.jsonl
→ BFF read-only view model
→ RankerReplayPanel
```

No model is loaded in Replay. The panel always shows
`Recorded Experiment Replay — Not a Live Model Call`.

## Isolation / safety

- BFF binds `127.0.0.1`.
- Browser never sends `user_id`, `user_key` or confirmation flags; MCP server binds the demo
  identity at process start.
- No GPU tokens, SSH keys or model endpoints are exposed to the browser.
- **Known demo limitation:** the two fixed MCP profiles are process-level memory namespaces;
  M9 state is not per browser session. The UI notice says memory writes are local process memory
  and reset when the BFF restarts. This is documented rather than hidden; production
  multi-tenant session isolation remains future work.
- Reset session clears turns/pending clarification only; it does not erase M9 history.
- `Memory OFF/ON` in the ranker panel is recorded replay, not a destructive live memory toggle.
