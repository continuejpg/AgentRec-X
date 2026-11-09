# AgentRec-X Web Demo — Shared Contract (Lead-owned)

> Status: INTERNAL BUILD CONTRACT. This file is the single source of truth shared by the
> BFF and frontend owners. The user-facing report is `WEB_DEMO_REPORT.md`.
> Code truth wins over docs; if this contract and the existing P17 tools disagree, follow the
> existing tool payload and tell the Lead.

## 0. Non-negotiable boundaries

- **MCP server is the only tool boundary.** The BFF may call only the 7 P17 MCP tools through
  the official `mcp.Client` + stdio long-lived connection. It must not import
  `recommendation.*` core modules or read the catalogue directly.
- **No fake data.** Every product, evidence span, preference and gate decision must come from a
  real MCP tool result. No mock products, no invented price/rating/stock/brand, no fabricated
  success text. UNKNOWN stays UNKNOWN.
- **No re-ranking.** The BFF formats payloads; it must not reorder, score, filter or "improve"
  recommendation/comparison results. Gate action/status is passed through verbatim.
- **Identity is server-bound.** Two demo profiles:
  - `profile_0049` → MCP server `--user-key p14e2_lite_0049`, read-only.
  - `profile_0057` → MCP server `--user-key p14e2_lite_0057`, write-enabled only when
    `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1`.
  Frontend receives only `profile_id`; it never sends `user_id`/`user_key`/`confirmed_by_user`.
- **No new runtime/model.** No LLM calls, no Qwen, no GPU, no retraining, no re-indexing.
  Qwen is replay-only.
- **Local demo only.** Bind 127.0.0.1; no public deployment/auth claims.

## 1. Process topology

```
Browser (Windows, Chrome/Edge)
  → Next.js 16 dev server (Windows, 127.0.0.1:3000)
      → HTTP + SSE (direct fetch, CORS allowed for localhost:3000)
  → FastAPI BFF (WSL Ubuntu-22.04, .venv/bin/python, 127.0.0.1:8787)
      → two long-lived MCP clients (official mcp.Client, stdio)
          → MCP server A: p14e2_lite_0049, --search-mode bge, no writes
          → MCP server B: p14e2_lite_0057, --search-mode bge, writes gated by env
              → existing AgentRec-X catalogue / BGE / planner / Gate / M9 / P15 / P16
```

- MCP server cold start is ~35–45 s (catalogue + BGE). The page must render before readiness
  and show a real initialization status. Do not start a server per request.
- Default search mode is `bge`; `WEB_DEMO_SEARCH_MODE=bm25` is allowed for fast dev/tests only.
- The BFF starts, health-checks and owns both MCP processes; it restarts them only through the
  explicit `/api/demo/restart` endpoint or process exit handling. Graceful shutdown closes clients.
- Never write MCP stdout; logs go to stderr and `runs/web_demo/logs/`.

## 2. HTTP API (BFF, base `http://127.0.0.1:8787`)

All request bodies are JSON. Pydantic request models use `extra="forbid"`.
Session ids are opaque (`web-<hex>`); every endpoint except health/config/replay needs a valid
session id.

### GET /api/health
```json
{
  "status": "starting|ready|degraded|error",
  "ready": false,
  "started_at": "2026-11-09T10:00:00Z",
  "uptime_s": 12.3,
  "catalogue_size": 76263,
  "servers": [
    {"profile_id":"profile_0049","user_key":"p14e2_lite_0049","status":"starting|ready|error",
     "search_mode":"bge","writes_enabled":false,"startup_ms":null,"error":null,"tools":null}
  ],
  "modes": {
    "cpu_live": {"available": false, "label":"CPU Live Mode",
                 "detail":"BGE dense retrieval + BM25 fallback + Rule Planner + V0 Scorer + P15/P16 + P17 MCP"},
    "qwen_replay": {"available": true, "label":"Qwen Replay Mode",
                    "badge":"Historical Replay — Not a Live Model Call"},
    "qwen_live": {"available": false, "label":"Qwen Live Mode",
                  "disabled_reason":"No live Qwen service detected (P18)"}
  },
  "memory_writes_enabled": true
}
```

### GET /api/demo/config
Profiles, modes, limits, verified comparison ids and notices. Frontend must use this list for
the profile selector; it must not invent a profile.
```json
{
  "profiles": [
    {"profile_id":"profile_0049","label":"Demo A · Read-only","description":"...",
     "user_key":"p14e2_lite_0049","memory_writes":false,"search_mode":"bge"},
    {"profile_id":"profile_0057","label":"Demo B · Memory write","description":"...",
     "user_key":"p14e2_lite_0057","memory_writes":true,"search_mode":"bge"}
  ],
  "default_profile_id": "profile_0049",
  "limits": {"search_top_k": 10, "compare_min": 2, "compare_max": 5, "clarification_ttl_s": 1800},
  "verified_comparison_ids": ["B004K6R43G", "B00008VF63"],
  "modes": { "...": "same as /health" },
  "notices": ["Local CPU demo; catalogue facts only; no live price/stock/ratings."]
}
```

### Session
- `POST /api/sessions` body `{"profile_id": "profile_0049"}` → `SessionView`
- `GET /api/sessions?limit=20` → `{"sessions":[SessionSummary]}`
- `GET /api/sessions/{session_id}` → `SessionView`
- `POST /api/sessions/{session_id}/reset` → `SessionView` (clears turns/pending clarification;
  does not clear M9 memory — document that backend restart resets memory)
- `GET /api/preferences?session_id=...` → `PreferenceState`

`SessionView`
```json
{
  "session_id": "web-1a2b3c4d",
  "profile_id": "profile_0049",
  "title": "New shopping session",
  "created_at": "...", "updated_at": "...",
  "turns": [TurnView],
  "pending_clarification": null
}
```
`SessionSummary` = session_id/profile_id/title/created_at/updated_at/turn_count.
`TurnView` fields: `turn_id`, `user_text`, `assistant_text`, `tool_activity` (ToolStatus[]),
`products` (ProductsPayload|null), `comparison` (ComparisonPayload|null),
`clarification` (ClarificationPayload|null), `memory_update` (MemoryUpdatePayload|null),
`error` (ErrorPayload|null), `created_at`.

`PreferenceState`
```json
{
  "profile_id":"profile_0049","user_key":"p14e2_lite_0049",
  "history_status":"active_preferences",
  "active":[{"memory_id":"...","value":"wireless","polarity":"prefer","kind":"feature",
             "status":"active","source_text":"I also prefer wireless.",
             "source_turn_id":"s1t0","last_updated":"2026-..."}],
  "inactive":[],
  "audit_source":"M9 PreferenceMemoryService via HistoryStateReader",
  "fetched_at":"..."
}
```
Only real `active_preferences` may be labelled active. Withdrawn/inactive entries appear only
under "Inactive / audit history".

### POST /api/chat/stream  (SSE)
Body: `{"session_id":"...","message":"...","selected_product_ids":["..."]}` (selected ids optional).
Response: `text/event-stream`. Events are `event: <type>\ndata: <json>\n\n`; heartbeat comment
`: keep-alive\n\n` every 5 s while a tool is running. All payloads are view models, never raw MCP JSON.

Event types (exact names):
- `tool_status` — `{tool, label, status:"running|ok|error", detail?, latency_ms?}`;
  labels: Search/Recommend/Product Info/Compare/Clarification/Memory/Evidence phases.
- `text` — `{text, final?: true}` assistant narrative (plain text; no markdown requirement).
- `products` — ProductsPayload (below).
- `evidence` — `{product_id, evidence:[EvidenceItem], unknown_fields:[UnknownField], source_record}`.
- `comparison` — ComparisonPayload (only when user asks to compare selected items).
- `clarification` — ClarificationPayload.
- `memory_update` — MemoryUpdatePayload.
- `done` — `{turn_id, stop_reason:"finalized|awaiting_clarification|error|comparison", total_ms}`.
- `error` — `{code, message, tool?, detail?}`. Never emit fake success after an error.

Chat flow (deterministic demo controller; no LLM):
1. Search the real catalogue (`search_products`, top_k=10, server search mode).
2. Enrich top search hits with `get_product_info` (batch; real evidence + UNKNOWN) to build cards.
3. Call `recommend_products`. If `status=="awaiting_clarification"` or gate.action is
   `NEED_USER_CLARIFICATION`/`ASK_CLARIFICATION`: emit search products (real), then call
   `request_preference_clarification` and emit `clarification`; stop with
   `stop_reason="awaiting_clarification"`.
4. Otherwise emit `products` with search cards + recommended card + Gate;
   emit `evidence` for the selected product; emit deterministic narrative text; `done`.
5. If the message is an explicit comparison request and `selected_product_ids` has 2–5 ids,
   call `compare_products` and emit `comparison` (instead of a new search).
Narrative must state: catalogue count, search mode, gate action, selected item, UNKNOWN fields,
and that catalogue evidence is not a quality/stock/price claim. No "best" claims beyond Gate.

### POST /api/products/details
Body `{"session_id":"...","product_ref":"B004K6R43G"}` → `ProductDetailPayload`.
Calls only `get_product_info`; maps real `product_info` facts, `product_evidence` spans,
`unknown_fields`, `source_record`. Echoes real values; no filling of unknowns.

### POST /api/products/compare
Body `{"session_id":"...","product_refs":["B004K6R43G","B00008VF63"]}` (2–5) →
`ComparisonPayload`. Calls only `compare_products`. Invalid ids/namespace/role must return
HTTP 400/422 with `{code, message, detail}` from the real MCP error. No mock fallback.

### POST /api/clarification/reply
Body `{"session_id":"...","clarification_id":"...","message":"I prefer noise cancelling and wireless."}`
→
```json
{
  "turn_id":"web-...-r1",
  "memory_update": {"added":2,"superseded":0,"removed":0,"skipped_duplicates":0,
                    "already_processed":false,"confirmation_source":"web_ui_explicit_user_reply"},
  "preferences": PreferenceState,
  "recommendation": ProductsPayload | null,
  "steps": [{"tool":"submit_clarification_reply","status":"ok","latency_ms":0.43}],
  "clarification_id":"...",
  "message": "...",
  "error": null
}
```
Rules:
- Validate the pending clarification belongs to this session; otherwise 409/400.
- The BFF sets `confirmed_by_user=true` only because a real HTTP POST body arrived; the frontend
  never supplies that flag. Use `confirmation_source="web_ui_explicit_user_reply"` (additive
  extension to the MCP tool Literal; `explicit_test_user_reply` and `trusted_local_client` stay valid).
- If the MCP server has writes disabled, return the real `memory_writes_disabled` error (409/403);
  do not show a success.
- On success, call `get_active_preferences`, emit/persist `memory_update` + refreshed preferences,
  then continue the pending query through `recommend_products` and return the real products.
- `turn_id` must be stable per reply attempt (idempotency); reuse on retry.

### GET /api/replay
Reads `runs/p18/qwen_replay.json` (existing real saved Qwen trajectories) and returns
`{"mode":"Qwen_Agent_Replay","replay":true,"live_call":false,
  "badge":"Historical Replay — Not a Live Model Call","examples":[...]}`.
No live Qwen endpoint exists. The frontend must visibly label replay; never present it as live.
The saved file is passed through as data: some trajectories are final-answer-only and have
`terminal: null` (or no `terminal` field) while still carrying `final_answer`. The frontend must
render those honestly as "terminal not recorded / final answer only" and must not crash or invent
a `selected_candidate_id`/`ask_question` for them.

## 3. View models (all mapped from real MCP payloads)

`UnknownField` = `{attribute, reason}`.
`EvidenceItem` = `{evidence_id, raw_field, span_start, span_end, span_text, raw_excerpt,
attribute, verification_status, source_record}`.

`ProductsPayload`
```json
{
  "query":"...", "search_mode":"bge", "score_kind":"bge_dense", "catalogue_size":76263,
  "items":[ProductCard], "recommended_product_id":"B004K6R43G"|null,
  "gate": {"action":"ALLOW_FINALIZE","reason_code":"evidence_sufficient","reason":"...",
           "selected_candidate_id":"...","unknown_fields":["over_ear"],"product_role":"main",
           "candidate_presentable_ids":["..."]} | null
}
```
`ProductCard`
```json
{
  "product_id":"B004K6R43G", "namespaced_id":"amazon_electronics_c4:B004K6R43G",
  "title":"...", "product_type":"earbuds", "form_factor":"in_ear", "product_role":"main",
  "source":"search|recommendation",
  "retrieval_source":"BGE dense retrieval|BM25 fallback|catalogue direct",
  "match_status":"search_result|recommended|eligible_candidate",
  "recommended": false,
  "price":{"status":"UNKNOWN","display":"Price unavailable / UNKNOWN",
           "reason":"no reliable numeric price in the catalogue record"},
  "attributes":[{"attribute":"in_ear","status":"present","value":"true",
                 "evidence_id":null,"raw_field":null,"span_text":null}],
  "unknown_fields":[{"attribute":"brand","reason":"..."}],
  "evidence_available": true,
  "image_url": null,
  "source_record":"data/p13r/electronics/catalog_meta.jsonl:amazon_electronics_c4:B004K6R43G"
}
```
- `image_url` stays null: the MCP catalogue snapshot exposes no image URL; the UI must render a
  neutral placeholder. Never hotlink random product photos.
- `attributes` = verifiable facts only (from `product_info.facts` with status `present` and/or
  evidence spans). Do not synthesize values from titles except quoting the real title span.
- `price` must be UNKNOWN unless a real value is returned; display text is fixed.

`ProductDetailPayload`
```json
{
  "product_id":"...", "namespaced_id":"...", "title":"...", "product_type":"...",
  "form_factor":"...", "product_role":null,
  "facts":[{"attribute":"title|price|brand|material|stock|warranty","status":"present|unknown|unsupported",
            "value":null,"reason":"...","source":"...","verification":"..."}],
  "evidence":[EvidenceItem],
  "unknown_fields":[UnknownField],
  "raw_fields":{"title":[...],"description":[...],"categories":[...],"features":[...]},
  "source_record":"...", "evidence_available":true,
  "notes":["catalogue facts only; UNKNOWN not filled from model knowledge"]
}
```
The UI renders "Catalogue facts" separately from "Your active preferences"; no automated P15
check may be described as human semantic review (`semantic_truth_verified=false`).

`ComparisonPayload`
```json
{
  "status":"comparison_ready",
  "products":[{"product_id":"...","title":"...","product_type":"...","role_status":"MATCH",
               "facts":[...],"unknown_fields":[UnknownField]}],
  "dimensions":[{"attribute":"price","cells":[{"product_id":"B004K6R43G",
                 "status":"unknown","value":null,"evidence_ids":[]}]}],
  "preference_relations":[{"memory_id":"...","value":"wireless","polarity":"prefer",
                            "relation":"not_evidenced","observed_in_catalogue_text":false,
                            "source_text":"I also prefer wireless.",
                            "per_product":{"B004K6R43G":"not_evidenced"}}],
  "conflicts":[],
  "unknown_fields":[{"attribute":"price","products":["B004K6R43G","B00008VF63"]}],
  "claims_summary":{"valid":16,"rejected":0,"unknown":12},
  "conditional_recommendation":{"supported":null,"text":null|"..."},
  "final_answer":"...",
  "evidence_sources":["data/p13r/electronics/catalog_meta.jsonl"],
  "notes":["catalogue evidence only; no real-world quality claim"]
}
```
Do not recompute MATCH/VIOLATION/UNKNOWN; map from P16. `conditional_best_supported` and
`final_answer` pass through unchanged. `conflicts` may be empty when P16 reports none.

`ClarificationPayload`
```json
{"clarification_id":"clar_...","question":"What matters most for this purchase ...",
 "reason":"missing_key_need","expires_in_seconds":1800,"memory_write_enabled":true,
 "quick_options":[]}
```
Only add `quick_options` when the real question supports fixed choices; the current P16 question
is free-text, so it must be `[]`.

`MemoryUpdatePayload`
```json
{"added":2,"superseded":0,"removed":0,"skipped_duplicates":0,"already_processed":false,
 "confirmation_source":"web_ui_explicit_user_reply","preferences":PreferenceState}
```

## 4. BFF module layout (`web_demo/`)

```
web_demo/__init__.py
web_demo/config.py        # paths, profiles, env settings, replay path
web_demo/mcp_manager.py   # long-lived mcp.Client per profile; readiness, lock, timeout, restart,
                          # stderr log path, graceful shutdown
web_demo/sessions.py      # in-memory session/turn store; id generation; pending clarification
web_demo/viewmodels.py    # pure mapping functions (payload -> view model), no I/O
web_demo/demo_flow.py     # deterministic event generators for chat / compare / details / reply
web_demo/app.py           # FastAPI app, CORS, routes, lifespan; SSE formatting
web_demo/__main__.py      # uvicorn.run(app, host=127.0.0.1, port=8787)
web_demo/README.md        # API + run + known behaviour
```
- No new Python dependency needed: `mcp`, `fastapi`, `uvicorn` are already in the WSL venv.
- Logs: `runs/web_demo/logs/bff.log`, `mcp-<profile>.log` (create dirs).
- Config env: `AGENTRECX_WEB_DEMO_PORT` (8787), `AGENTRECX_WEB_DEMO_SEARCH_MODE`
  (`bge` default, `bm25` dev), `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` (`0` default; launcher
  for the demo sets `1`), `AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S` (180).
- Minimal allowed edit outside `web_demo/`: add `"web_ui_explicit_user_reply"` to the
  `confirmation_source` Literal in `recommendation/mcp/server.py`; keep old values and old tests.
  No other core change.

## 5. Frontend (`web/`)

Stack: Next.js 16 (App Router), React 19, TypeScript strict, Tailwind CSS 4, Node 24 on Windows.
Run: `cmd /c "pnpm --dir web dev"` (PowerShell blocks pnpm.ps1). Dev port 3000.

Required layout:
- Desktop (≥1280): left sidebar (brand, new conversation, recent conversations, demo profile,
  active/inactive preferences, clear session); center chat; right contextual panel
  (details/evidence/comparison) collapsible.
- Mobile (<768): single column; sidebar/preferences/details/comparison in Sheets/Drawers;
  composer fixed at bottom; product cards adapt to narrow width.
- Header: AgentRec-X brand, current profile, mode badge (CPU Live default; Qwen Replay clearly
  labelled `Historical Replay — Not a Live Model Call`; Qwen Live disabled), readiness status.
- Chat: user/assistant bubbles, tool-activity timeline with real tool labels/statuses/latencies,
  product cards, evidence, comparison, clarification card, error state, suggested real actions.
- Product card: neutral placeholder image (own SVG; no random photos), title, ID/namespace,
  type/role, retrieval source, match status, verifiable attributes, UNKNOWN, evidence available,
  `Price unavailable / UNKNOWN`, buttons: View details / View evidence / Select for comparison.
- Detail/evidence drawer: real facts, raw field/span/span text, source record, UNKNOWN, and a
  separate "Your active preferences" block. No human-review claims.
- Compare: select 2–5, table of dimensions, MATCH/VIOLATION/UNKNOWN, preference relations,
  conflicts, evidence sources, conditional recommendation; invalid selection shows real error.
- Clarification: question, free-text reply, submit; progress stages; after success refresh
  preferences from MCP, show real memory update, continue the pending query.
- No cart/checkout/payment. No invented price/rating/stock.

Styling: modern e-commerce assistant, light theme, tokenized CSS variables; reuse/adapt the
Anthropic `web-shared`/`storefront-web` components listed in `third_party/UPSTREAM_REUSE_GUIDE.md`
with the upstream header retained and a modification note. Do not copy product images/brand
assets. Placeholder art is ours.

Component/test requirements: `pnpm build` and `pnpm exec tsc --noEmit` pass; Playwright E2E
added later by the E2E owner under `web/e2e/` and `web/playwright.config.ts`.

## 6. Testing & acceptance

BFF:
- unit tests with a fake MCP manager: session isolation, request validation, SSE event order,
  tool error mapping, no confirmed_by_user from client body, writes-disabled error, replay.
- one live integration test (BM25 for speed, marked `slow`) covering: connect, tool discovery,
  search (catalogue_size 76263), recommend, compare, clarification + `web_ui_explicit_user_reply`,
  `get_active_preferences` change. Do not run full pytest; targeted files only.
- Runtime smoke via HTTP: health → session → chat SSE → details → compare → clarification reply
  (profile 0057, writes enabled) → prefs refresh. Save transcript to `runs/web_demo/`.

Frontend:
- `pnpm build`; TypeScript; component behavior; responsive; real SSE; error/loading/empty.
E2E (separate owner): real browser Chrome (system channel), 3 scenarios, desktop 1440×900 and
mobile 390×844 screenshots under `runs/web_demo/screenshots/`; trace under `runs/web_demo/traces/`.

Acceptance for the whole demo:
1. `GET /api/health` shows real MCP readiness for both profiles.
2. Scenario 1: natural-language search returns real catalogue cards; evidence drawer shows real
   spans + UNKNOWN; no price invention.
3. Scenario 2: compare the two verified ids (B004K6R43G, B00008VF63) returns the real P16
   comparison; invalid id returns a real error.
4. Scenario 3: vague query → real NEED_USER_CLARIFICATION → user reply → M9 memory update
   (added 2 for the P18 reply) → preferences panel refresh → recommendation continues.
5. CPU Live vs Qwen Replay visibly distinct; Qwen Live disabled.
6. One local command starts BFF + frontend; docs and notices complete; no git commit/push.

## 7. Environment quick reference

- Repo (Windows): `D:\IT\CODE\PROJECTION\agentrec\X`; WSL path `/mnt/d/IT/CODE/PROJECTION/agentrec/X`.
- Python: WSL `.venv/bin/python` (3.10.12); run `wsl -d Ubuntu-22.04 --cd /mnt/d/... -- .venv/bin/python ...`.
- Node: Windows `v24.14.1`, npm 11.11.0, pnpm 12.4.1 (invoke via `cmd /c`).
- MCP client/server: `mcp==2.3.0`; startup ~35–45 s; do not restart per request.
- Reference clones (read-only, outside repo): `D:\IT\CODE\PROJECTION\agentrec\_refs\<repo>`.
- Existing worktree has 332 untracked + many modified files from earlier milestones: do not
  revert/overwrite them; only add new files in the scopes listed in the team task.
