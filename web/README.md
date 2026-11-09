# AgentRec-X Web Demo — frontend (`web/`)

Local shopping-assistant UI for AgentRec-X. It renders **only** structured view models from the
Web Demo BFF (`web_demo/`, FastAPI on `127.0.0.1:8787`); it never parses the P17 raw JSON, never
re-ranks products, and never invents prices, ratings, stock, evidence or preferences.

- Contract source of truth: [`../docs/WEB_DEMO_CONTRACT.md`](../docs/WEB_DEMO_CONTRACT.md)
- Demo guide: [`../DEMO_GUIDE.md`](../DEMO_GUIDE.md)
- Stack: Next.js 16 (App Router) · React 19 · TypeScript strict · Tailwind CSS 4 · Node 24 on Windows

## Run it

```cmd
:: 1. start the BFF (WSL, from the repo root) — writes enabled for Demo B
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- bash scripts/run_web_demo_bff.sh

:: 2. install + start the frontend (Windows)
cmd /c "pnpm --dir web install"
cmd /c "pnpm --dir web dev"
```

Open <http://localhost:3000>. The page renders immediately; the composer unlocks when
`/api/health` reports both MCP profiles ready (cold start is normally 35–45 s and the UI shows the
real per-server status while it happens).

The BFF base URL defaults to `http://127.0.0.1:8787`. Override it with
`NEXT_PUBLIC_BFF_URL` (e.g. `set NEXT_PUBLIC_BFF_URL=http://127.0.0.1:8790` before `pnpm dev`).

Other scripts:

```cmd
cmd /c "pnpm --dir web build"              :: production build
cmd /c "pnpm --dir web exec tsc --noEmit"  :: strict type check
cmd /c "pnpm --dir web start"              :: serve the production build on :3000
```

## What the UI does

- **Desktop ≥1280 px**: sidebar (brand, new conversation, recent conversations, demo profile,
  active preferences, clear session) · chat column · collapsible contextual panel with
  **Details / Evidence / Comparison** tabs.
- **Mobile / tablet <1280 px**: single column; sidebar and the contextual panel are drawers;
  composer is pinned to the bottom; product cards turn into a horizontal snap carousel.
- **Chat**: user/assistant bubbles, a real tool-activity timeline built from `tool_status`
  events (phase copy + tool name + real latency), streaming narrative, product cards, Gate
  decision, clarification card, memory-update card, comparison card, error and interrupted
  states, and suggested *real* actions only.
- **Product card**: self-drawn neutral placeholder SVG (the catalogue snapshot has no image
  URL), real title/ID/namespace/type/role, retrieval source, match status, verifiable
  attributes, UNKNOWN fields with reasons, `Price unavailable / UNKNOWN`, evidence availability
  and the 2–5 comparison selection.
- **Details / evidence**: real facts, raw field + span + span text + source record, UNKNOWN
  fields, and a separate **Your active preferences** block (M9 memory is not product evidence).
  P15 `verification_status` is presented as automated checks — no human semantic review is
  claimed.
- **Comparison**: P16 `ComparisonPayload` rendered verbatim — dimensions/cells, preference
  relations, conflicts, UNKNOWN fields, claims summary, conditional recommendation, evidence
  sources and notes. Invalid references show the real BFF/MCP error; there is no mock fallback.
- **Clarification**: real question, free-text reply (quick options only when the backend sends
  them), real memory-update progress, refreshed preferences and the continued recommendation.
  On the read-only profile the reply is refused by the backend, so the card offers to switch to
  the write-enabled Demo profile and retry the same query.
- **Modes**: CPU Live is the default live pipeline. Qwen Replay opens the saved
  `runs/p18/qwen_replay.json` trajectories through `GET /api/replay`, loudly labelled
  `Historical Replay — Not a Live Model Call`; Qwen Live is disabled with the backend's reason.
- **Readiness**: real `/api/health` polling with per-MCP-server status, startup time, catalogue
  size and memory-write state; the page stays interactive while services start.
- **Session state**: sessions are backend in-memory; only the session id is persisted in
  `localStorage`, so a refresh restores the conversation. `New conversation` creates a new
  session; `Clear session` resets turns/pending clarification (M9 memory is not cleared).

Out of scope by contract: cart, checkout, payment, auth, public deployment, live Qwen calls.

## Source layout

```
app/                Next.js App Router entry (layout, page, global CSS + adapted base.css)
components/         DemoApp orchestrator + chat, product, evidence, comparison, replay UI
lib/                BFF contract types, fetch/SSE client, formatting, browser turn state
public/             favicon (self-drawn; no third-party product images)
```

## Data & error rules

- Every product, evidence span, preference and Gate decision comes from a real MCP tool call made
  by the BFF. UNKNOWN stays UNKNOWN; no price/rating/stock is fabricated.
- `POST /api/chat/stream` is consumed with `fetch` + `ReadableStream` (POST cannot use
  `EventSource`); the parser handles `event:`/`data:` frames, `: keep-alive` comments, aborts
  (`AbortController`), timeouts and interrupted streams.
- Real errors are surfaced with their BFF `code`/`message`/`detail`; the UI never shows a fake
  success after an error.
- The frontend never sends identities or confirmation flags (`user_id`, `user_key`,
  `confirmed_by_user`); it only sends session ids, message text and opaque product references.

## Licensing / upstream reuse

Parts of this UI are adapted from Anthropic's Apache-2.0 `commerce-agents` repository
(`examples/web-shared/*`, `examples/retail/storefront-web/*`, commit `fd4d592`) — every adapted
file keeps the `Copyright 2026 Anthropic PBC` + `SPDX-License-Identifier: Apache-2.0` header with a
modification note. The per-file inventory lives with the project notices
(`THIRD_PARTY_NOTICES.md` / `third_party/`). Placeholder product art is drawn in this project; no
storefront images or brand assets are copied. `shop-gym` / `PAHF` were design references only;
`techjam-*` / `coveo-*` are unlicensed and nothing was copied from them.

## Testing status

- `pnpm --dir web exec tsc --noEmit` and `pnpm --dir web build` pass.
- Browser E2E (Playwright, `web/e2e/**`, `web/playwright.config.ts`) is owned by the later E2E
  task and is intentionally not created here. Manual browser verification against the real BFF
  covers: search + evidence + details, 2–5 product comparison (and the real error path), and
  clarification → memory update → refreshed preferences → continued recommendation.
- This is a local demo only. It does not establish recommendation quality, real-user satisfaction
  or production security.
