# AgentRec-X Web Demo BFF (`web_demo/`)

Local FastAPI backend-for-frontend for the AgentRec-X Web Demo. It owns two long-lived
official-MCP stdio clients (one per fixed demo profile) and exposes the HTTP + SSE contract in
[`docs/WEB_DEMO_CONTRACT.md`](../docs/WEB_DEMO_CONTRACT.md).

```text
Browser (Windows)
  → Next.js dev server 127.0.0.1:3000
  → FastAPI BFF 127.0.0.1:8787          (WSL Ubuntu-22.04, .venv/bin/python)
      → mcp.Client A stdio → recommendation.mcp.server --user-key p14e2_lite_0049 (read-only)
      → mcp.Client B stdio → recommendation.mcp.server --user-key p14e2_lite_0057 (writes gated)
```

**No fake data.** Every product, evidence span, UNKNOWN field, preference and Gate decision is a
real payload from the P17 MCP tools. The BFF never imports `recommendation.*` core modules, never
re-ranks/reorders/filters tool results, and never invents price/rating/stock/brand.

**CPU default; optional Qwen Base Live.** CPU mode makes no LLM call. P20 can optionally connect to
an **existing** Qwen3.5-9B Base model service when explicitly configured and healthy; a missing
service leaves Qwen Live Unavailable and never falls back to Rule while claiming Qwen. Qwen Agent
Replay (`GET /api/replay`, `live_call=false`) remains historical only. Local demo only: binds
`127.0.0.1`.

## Run

WSL Ubuntu-22.04, from the repository root:

```bash
# one command: starts both MCP servers in parallel and serves the BFF
bash scripts/run_web_demo_bff.sh
```

The launcher defaults to `AGENTRECX_WEB_DEMO_SEARCH_MODE=bge` and
`AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1` (the BFF itself still defaults writes to **0**), writes
`runs/web_demo/bff.pid`, and tees console output to `runs/web_demo/logs/bff.log`.
For a fast dev loop: `AGENTRECX_WEB_DEMO_SEARCH_MODE=bm25 bash scripts/run_web_demo_bff.sh`.

Direct run (no launcher):

```bash
.venv/bin/python -m web_demo
```

Cold start is real: each MCP server loads the catalogue (76,263 products) plus BM25 or BGE, about
35–45 s in BGE mode. `GET /api/health` answers immediately with per-server `starting` status;
the page must render before readiness. BGE is the demo default; `bm25` is for fast dev/tests only.

| Env var | Default (BFF) | Meaning |
|---|---|---|
| `AGENTRECX_WEB_DEMO_PORT` | `8787` | HTTP port (host is always `127.0.0.1`) |
| `AGENTRECX_WEB_DEMO_SEARCH_MODE` | `bge` | `bge` (demo) or `bm25` (fast dev/tests) |
| `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` | `0` | `1` enables writes for profile_0057's MCP server |
| `AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S` | `180` | per-tool-call timeout (startup uses the same ceiling) |
| `AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG` | unset | launcher-only: BFF skips its own bff.log handler because the launcher tees console output there |
| `QWEN_LIVE_ENABLED` | `0` | explicit opt-in for the optional real Qwen Base Live mode |
| `QWEN_MODEL_ENDPOINT` | unset | existing model server base URL; never exposed to the browser/logs |
| `QWEN_MODEL_NAME` | `Qwen3.5-9B-Base` | informational model label |
| `QWEN_LIVE_TIMEOUT_S` / `QWEN_LIVE_HEALTH_TIMEOUT_S` | `180` / `60` | planner call / real PlannerAction health timeout |
| `QWEN_LIVE_TURN_TIMEOUT_S` | `600` | bound for the complete multi-step Qwen turn |
| `QWEN_LIVE_MAX_STEPS` | `8` | planner step budget (server also hard-caps) |

Qwen Live is available only after the BFF's real `POST /plan` health probe returns a valid
PlannerAction. A configured-but-unreachable service keeps CPU Live and Qwen Replay working and
shows Qwen Live as Unavailable; the BFF does not start/stop AutoDL/GPU, create an SSH tunnel,
download/quantize a model, or silently fall back to Rule. When configured, the MCP server exposes
the original 7 tools plus the optional 8th `run_agent_turn`. See
[`../docs/QWEN_LIVE_CONTRACT.md`](../docs/QWEN_LIVE_CONTRACT.md) and
[`../QWEN_LIVE_WEB_REPORT.md`](../docs/qwen/QWEN_LIVE_WEB_REPORT.md). On 2026-11-09 this path was validated
with real Qwen3.5-9B Base inference, real MCP/executor/Gate/P15 execution and two real browser
turns on an RTX 4090; the temporary model server/tunnel are stopped afterwards. The model did not
choose ASK in the tested prompts, so that continuation branch is not claimed.

Stop a launcher-started BFF with `kill "$(cat runs/web_demo/bff.pid)"` (graceful: MCP clients are
closed in the FastAPI lifespan). From Windows PowerShell use
`wsl -d Ubuntu-22.04 -- bash -lc 'kill "$(cat /mnt/d/IT/CODE/PROJECTION/agentrec/X/runs/web_demo/bff.pid)"'`.

## Endpoints

All request bodies are JSON; request models use `extra="forbid"` so a browser can never inject
identity or confirmation (`user_id`, `user_key`, `confirmed_by_user`, `confirmation_source` → 422).

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | status, real per-server status/startup_ms/tool list, catalogue_size, modes |
| GET | `/api/demo/config` | profiles, default profile, limits, verified comparison ids, notices |
| POST | `/api/demo/restart` | `{"profile_id": optional}` → schedules a real MCP server restart |
| POST | `/api/sessions` | `{"profile_id": "profile_0049"}` |
| GET | `/api/sessions?limit=20` | recent sessions |
| GET | `/api/sessions/{id}` | `SessionView` (turns + pending clarification) |
| POST | `/api/sessions/{id}/reset` | clears turns/pending clarification; M9 memory is untouched |
| GET | `/api/preferences?session_id=` | real M9 active/inactive preference state |
| POST | `/api/chat/stream` | SSE turn: `text/event-stream` |
| POST | `/api/products/details` | `{"session_id","product_ref"}` → real P15 evidence + UNKNOWN |
| POST | `/api/products/compare` | `{"session_id","product_refs":[...2-5]}` → real P16 comparison |
| POST | `/api/clarification/reply` | persists a real user reply through M9, refreshes prefs, resumes query |
| GET | `/api/replay` | real saved P18 Qwen trajectories, clearly labelled historical replay |

SSE event names (exact): `tool_status`, `text`, `products`, `evidence`, `comparison`,
`clarification`, `memory_update`, `done`, `error`. While a tool runs, a `: keep-alive` comment is
sent every 5 s. `done.stop_reason` is one of `finalized|awaiting_clarification|error|comparison`;
`finalized` means the turn reached a terminal decision (including a real
`no_verified_candidate_in_scope`/`budget_exhausted` status — the narrative states the real status),
`error` is reserved for real MCP/transport failures.

Example:

```bash
curl -sN -X POST http://127.0.0.1:8787/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"web-xxxxxxxx","message":"wireless earbuds for working out"}'
```

## Chat controller (deterministic; no LLM)

1. `search_products` (`top_k=10`, server's real search mode).
2. `get_product_info` for the real search hits (real facts/evidence/UNKNOWN → cards).
3. `recommend_products`; if the Gate asks for clarification (`awaiting_clarification` /
   `ASK_CLARIFICATION` / `NEED_USER_CLARIFICATION`) the BFF emits real search products, then
   `request_preference_clarification`, the `clarification` event and
   `done.stop_reason="awaiting_clarification"`.
4. Otherwise it emits `products` (search cards + real recommended card + verbatim Gate), then
   `evidence` for the selected product, deterministic narrative text, and `done`.
5. An explicit comparison request with 2–5 `selected_product_ids` calls `compare_products` instead
   of a new search and emits `comparison`.

View models keep UNKNOWN as UNKNOWN: a card's `price` is `Price unavailable / UNKNOWN` unless the
catalogue really returns a price value; `image_url` is always `null` (the catalogue snapshot has no
image URL); Gate action/status/candidate ids pass through unchanged.

## Clarification reply and the memory-write boundary

`POST /api/clarification/reply` only works for the session's own pending clarification (otherwise
409). The BFF validates that writes are enabled on that profile's MCP server *before* any call and
returns the real `memory_writes_disabled` error (409) when they are not — never a fake success.

Because a real HTTP POST with user-authored text arrived, the BFF (not the browser) sends
`confirmed_by_user=true` and `confirmation_source="web_ui_explicit_user_reply"`. That value is an
additive extension to the MCP tool's `confirmation_source` Literal in
`recommendation/mcp/server.py`; `explicit_test_user_reply` and `trusted_local_client` remain valid
and all previous behaviour is preserved. The reply's `turn_id` is generated when the clarification
is issued and reused on retry, so the MCP-side write is idempotent (`already_processed=true`).

On success the BFF calls `get_active_preferences` (refresh), then resumes the pending query through
`recommend_products` and returns the real products + steps + memory update counts.

**Memory is process-local fixture memory:** writes live inside the MCP server process and reset when
the BFF/MCP server restarts. `reset` clears the conversation, not M9 memory.

## Logs and artifacts

- `runs/web_demo/logs/bff.log` — BFF/uvicorn console log (FileHandler direct run, or launcher tee).
- `runs/web_demo/logs/mcp-profile_0049.log` / `mcp-profile_0057.log` — the **real** MCP server
  stderr (catalogue/BGE/planner logs) captured through a `bash -c 'exec ... 2>>log'` wrapper; the
  MCP stdout is never written (it carries the stdio protocol).
- `runs/web_demo/bff.pid` — launcher PID file.
- `runs/web_demo/mcp_integration_transcript.json` — real BM25 integration-test transcript.
- `runs/web_demo/bff_smoke_*.json` — real HTTP smoke transcript (health → session → chat SSE →
  details → compare → clarification reply → preferences).

## Tests

```bash
# fast unit tests with a fake MCP manager (no server, no WSL cold start)
.venv/bin/python -m pytest -q tests/test_web_demo_unit.py

# one live MCP integration test (BM25, slow; starts two real servers)
AGENTRECX_RUN_MCP_INTEGRATION=1 \
  .venv/bin/python -m pytest -q tests/test_web_demo_integration.py
```

Unit coverage: session isolation/profile binding, `extra="forbid"` identity injection rejection,
SSE event order and payload mapping, cancellation leaves no dirty session state, real tool-error
mapping, writes-disabled reply error, clarification reply confirmation construction, replay flags.
Integration coverage: connect + 7-tool discovery, catalogue_size 76263, product evidence/UNKNOWN,
real recommendation Gate, P16 comparison + real error path, M9 memory update (2 added for the P18
reply, `web_ui_explicit_user_reply`), preference refresh, idempotent retry, read-only refusal.

## Known behaviour / limitations

- Single-process in-memory sessions and M9 fixture memory; BFF restart resets both.
- No auth/HTTPS/rate limiting: local trusted demo only, bind 127.0.0.1.
- BGE latency: server startup (catalogue + index) is ~35 s; the **first** `search_products` per
  MCP server also pays a real BGE encoder/encode cost (measured 18–34 s on this host), so the first
  chat turn of each profile is slow while later queries are fast (`recommend_products` ~150 ms hot).
  The SSE keep-alive comments keep the browser connection alive during this wait; the UI must show
  the tool timeline rather than a spinner-only state.
- `semantic_truth_verified=false`: evidence is P15 lexical/structural; no human semantic review.
- Qwen Live is unavailable (`modes.qwen_live.available=false`); Qwen Replay is labelled.
