# AgentRec-X Final Demo V2 — Technology Selection

## 1. Existing stack (inspected first)

The repository already contains a production-shaped local demo:

```text
web/                 Next.js 16 App Router / React 19 / TypeScript / Tailwind CSS 4
web_demo/            FastAPI BFF on 127.0.0.1:8787
recommendation/demo/ existing Agent, Gate, evidence, comparison and session modules
recommendation/mcp/  the real P17 MCP adapter over the existing stack
```

The existing `web/` already has:

- chat transcript and SSE streaming;
- product cards with UNKNOWN/evidence handling;
- real tool/timeline rows;
- clarification cards and M9 preference refresh;
- P15 details / P16 comparison;
- Qwen Replay and optional Qwen Live modes;
- no fake prices, ratings, stock or evidence.

**Decision: reuse this UI and BFF. Do not create a second chat app, React SPA, memory store or agent executor.**

No new UI framework, no new message store, no LangGraph migration, no new ranking implementation was added.

## 2. References studied

| Reference | What was studied | How it influenced this demo |
|---|---|---|
| [NVIDIA Retail Shopping Assistant](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant) | Conversational shopping layout, product result organization, separation of chat/retrieval/context, streaming assistant responses | Confirmed the chat-first + product shelf + context side panel layout. No LangGraph/services/cart copied. |
| [WOW-WILDLOTUS Shopping Copilot](https://github.com/Atirian-Chen/WOW-WILDLOTUS-Shopping-Copilot) | Chainlit chat, product cards/shelves, pipeline progress, node inspector, eval/diagnostic separation | Confirmed the key UX rule: **chat for users, separate inspector for agent internals**. No code copied. |
| [Chainlit Step docs](https://docs.chainlit.io/api-reference/step-class) / [Chainlit](https://docs.chainlit.io/) | Built-in steps, progress, session, actions, custom elements | Studied only. Chainlit is announced as archived/no longer maintained, so it is **not** introduced into a repo that already has a working Next.js BFF UI. |
| [Gradio Chatbot / Agents guide](https://www.gradio.app/guides/agents-and-tool-usage) | ChatInterface, Blocks, tool usage display, state, examples | Not selected: repo already has a richer maintained web UI. |
| [Open WebUI docs](https://docs.openwebui.com/) | Conversation organization, model selection, tool-call presentation, interaction details | Product-experience reference only; not installed. |
| [Arize Phoenix](https://arize.com/docs/phoenix/) / [OpenInference](https://arize-ai.github.io/openinference/) | Trace/tool execution/latency/retrieval observability | Reused the project's existing tool-status/timeline data instead of adding a second observability backend. |

Licences were checked at the reference level. No source file was copied from these projects. The existing `web/` already carries its own Apache-2.0 notices for adapted components; this phase adds no copied third-party code.

## 3. What is reused directly

- Existing `web/` chat shell, SSE client, message bubbles, product cards, evidence panel, comparison panel, sessions, readiness state and replay viewer.
- Existing `web_demo/` FastAPI BFF, session store, MCP manager, viewmodels and health/config endpoints.
- Existing `recommendation.mcp.server` tools (catalogue search, recommendation, M9 preferences, clarification, comparison).
- Existing M9 `PreferenceMemoryService` and its audit entries.
- Existing P100 recorded GPU predictions (rankers/Memory OFF-ON).
- Existing P100/P20 experiment artifacts for replay.

## 4. Minimal custom additions

Only thin adapters/view components were added:

1. **Memory timeline** (`web/components/MemoryTimeline.tsx`): renders M9 active/inactive entries with `logical_seq`, status, source turn, supersession/withdrawal. No new memory store.
2. **Memory Lab** (`web/components/MemoryLab.tsx`): explicit user-authored text to the **existing** M9 writer through a small
   `remember_preference` MCP adapter; it does not rerank by itself. This is demo-only and requires the write-enabled local profile.
3. **Ranker replay panel** (`web/components/RankerReplayPanel.tsx`): displays saved Phase 5.1 P100 GPU scores for Existing / ShopRanker Natural / Fixed Hybrid, with Memory OFF/ON recorded ranks. Clearly labelled **Recorded Experiment Replay — Not a Live Model Call**.
4. **Scenario cards** in the existing empty-chat state: three real prompts for Remember / Update-Negative / Withdraw.
5. BFF endpoint `/api/replay/rankers` and view-model: reads the saved P100 prediction JSONL; no model call.
6. BFF endpoint `/api/memory/remember` and MCP tool `remember_preference`: thin write adapter over the existing M9 service.

## 5. Why not Chainlit / Gradio / React rewrite

- The repo already has a working, type-checked, tested Next.js 16 UI with SSE, product cards, evidence, comparison, replay and accessibility states.
- Chainlit is officially archived; adding it would create a second chat framework and duplicate the existing BFF/UI.
- Gradio would replace a richer existing UI and require rewriting cards/trace/comparison.
- A new React SPA would rewrite session, SSE and message state for no user benefit.
- Phoenix/OpenInference would add a second tracing backend; the existing tool-status stream is the real trace source.

## 6. Live / Replay boundary

- **CPU Live**: real BFF → real MCP tools → real catalogue BGE, planner (Rule), Gate, M9, P15/P16.
- **Qwen Replay**: historical Qwen trajectories, read-only, explicitly labelled.
- **Qwen Live**: optional and shown only if the existing backend health probe says available; no silent fallback.
- **P100 Ranker Replay**: saved GPU predictions; no live ShopRanker call is fabricated.

The UI never claims that a saved model score is a live inference.
