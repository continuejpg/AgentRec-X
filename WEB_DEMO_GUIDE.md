# AgentRec-X Web Shopping Assistant — Local Demo Guide

> Status: local research demo. It runs on `127.0.0.1` only, has no public authentication and
> makes no production-quality claim. Everything you see comes from the real AgentRec-X
> catalogue/MCP tools — there is no mock product data.

## 1. What you get

A user-facing shopping assistant over the existing AgentRec-X stack:

```
Browser (Windows, Chrome/Edge)
  → Next.js 16 dev server               http://127.0.0.1:3000
  → FastAPI BFF (WSL Ubuntu-22.04)      http://127.0.0.1:8787
      → long-lived official mcp.Client (stdio, one per demo profile)
          → recommendation.mcp.server (P17)
              → real Electronics catalogue (76,263 products) / BGE + BM25
                Rule Planner + V0 + Role + Eligibility + Decision Gate
                P15 Evidence Bundle / P16 Comparison + Clarification
                M9 preference memory
```

- **Real search & recommendation**: BGE dense retrieval with BM25 fallback (server-configured),
  existing Rule Planner / Decision Gate.
- **Real evidence**: P15 catalogue spans (`raw_field`, `span`, `span_text`, `source_record`) and
  explicit `UNKNOWN` fields. The UI never fills price, brand, stock, ratings or material.
- **Real comparison**: P16 Shopping Comparison over 2–5 catalogue products.
- **Real clarification + memory**: a Gate clarification is answered by you in the browser; the
  reply is persisted through the existing M9 writer, then the preference panel refreshes and the
  recommendation resumes.
- **CPU Live vs Qwen Replay**: CPU Live is the only live mode. Qwen is shown only as saved P18
  trajectories, always labelled **Historical Replay — Not a Live Model Call**. Qwen Live is
  disabled (no live service).
- **No cart, checkout, payment, order or stock.** `Price unavailable / UNKNOWN` is the honest
  state for this catalogue snapshot.

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 + WSL2 (`Ubuntu-22.04`) | The back end runs in WSL; the front end runs on Windows. |
| WSL `.venv` at repo root | Python 3.10.12 with `mcp==2.3.0`, `fastapi`, `uvicorn`, `pydantic` (already used by P17/P18). |
| Local catalogue + BGE assets | `data/`, `models/`, `runs/p18/qwen_replay.json` must exist (as in the author's workspace). |
| Node.js + pnpm on Windows | Node 24 (v24.14.1 tested), pnpm 12.4.1. PowerShell may block `pnpm.ps1`; commands below use `cmd /c`. |
| Chrome or Edge | E2E uses the installed system Chrome (`channel: "chrome"`); no browser download is needed. |
| GPU | **Not needed.** The whole demo is CPU-only. |

No new Python package and no new frontend runtime dependency beyond the checked-in
`web/package.json` are required.

## 3. Start / stop (one command)

From the repo root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
```

This:

1. installs `web/node_modules` if missing (`pnpm --dir web install`);
2. starts the FastAPI BFF in WSL (BGE, two demo profiles, memory writes enabled for the demo
   launcher) with logs + PID under `runs/web_demo/`;
3. starts the Next.js frontend on Windows;
4. waits for the **real** `/api/health` readiness and prints per-server startup, then opens
   <http://127.0.0.1:3000>.

Cold start is real: each MCP server loads the catalogue + BGE and takes **~35–47 s** in this
environment (E2E measured 35.6 s / 35.5 s per server, 46.7 s from launcher to `/api/health`
ready); the page renders immediately and shows a real initialization banner. The **first BGE
query per server** also pays encoder warm-up (measured **~57–83 s** in the final E2E run on this
CPU host, load-dependent); subsequent searches are ~0.15–2.1 s. Tool activity and SSE keep-alives
make this visible instead of a silent spinner.

Both `http://127.0.0.1:3000` and `http://localhost:3000` work (`web/next.config.ts` allows both
loopback dev origins; Next.js 16 otherwise blocks dev resources opened through the other spelling).

Stop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1
```

Check readiness directly at any time:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/health | ConvertTo-Json -Depth 4
```

Useful flags: `-NoBrowser` (do not open a browser), `-SkipInstall`, `-BffPort`, `-FrontendPort`,
`-ReadyTimeoutSec`.

Logs:

- `runs/web_demo/logs/bff.log` — BFF requests/tool timings;
- `runs/web_demo/logs/bff.console.log` / `.err.log` — launcher console;
- `runs/web_demo/logs/frontend.console.log` / `.err.log` — Next.js;
- `runs/web_demo/logs/mcp-profile_0049.log`, `mcp-profile_0057.log` — real MCP server stderr
  (manifests, catalogue/BGE load, warnings).

## 4. Demo profiles and security boundary

| Profile | User key | MCP writes | Intended use |
|---|---|---|---|
| **Demo A · Read-only** (`profile_0049`) | `p14e2_lite_0049` | **disabled** | Search, recommendation, evidence, comparison. Clarification replies are refused by the read-only MCP server (real error shown). |
| **Demo B · Memory write** (`profile_0057`) | `p14e2_lite_0057` | enabled only when the **launcher** sets `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1` | Scenario 3: clarification → M9 update → refreshed active preferences → resumed recommendation. |

- The browser sends only `profile_id`; it never sends a user id, `confirmed_by_user` or
  `confirmation_source`. The BFF constructs the write confirmation **only** after a real HTTP
  POST carrying your typed reply, and labels the channel `web_ui_explicit_user_reply`.
- The MCP server binds the user at process start; there is no caller `user_id` parameter. Cross-
  profile memory access is impossible through the demo API.
- M9 memory in this demo is process memory backed by the P17 fixture: restarting the BFF
  (or the MCP server) resets it to the fixture. This is stated in the UI and is not a durable
  database claim.
- Localhost only; no auth/TLS/rate limiting. Do not expose this demo to a network.

## 5. Three real scenarios to try

### Scenario 1 — Search & recommend with evidence

1. Keep **Demo A · Read-only**.
2. Type: `I need wireless earbuds for working out.`
3. Watch the tool activity: `Searching products` (BGE) → `Loading product details` (P15) →
   `Validating constraints / recommendation` (Rule Planner + Gate).
4. Real product cards appear with ASIN/namespace, product type/form factor, retrieval source,
   evidence status, verifiable attributes and `UNKNOWN` fields; price always reads
   `Price unavailable / UNKNOWN` with a reason (the catalogue snapshot has no trusted price).
5. Click **View details / evidence** on a card: original title/description/categories spans,
   raw field + span + span text + source record, `UNKNOWN` attributes. Facts and your preferences
   are shown separately. P15 validation is labelled *automated*; it is not human semantic review.
6. The assistant text reports the real Gate action (e.g. `ALLOW_FINALIZE`) and says the catalogue
   evidence is not a quality/stock/price claim.

### Scenario 2 — Compare two verified products

1. Search `Audio Technica ATH-CKP300 Sport Fit Earbuds` and select **B004K6R43G** for comparison.
2. Search `Sony MDR-E828LP Fontopia Earbuds` and select **B00008VF63** (the P18-verified pair).
3. Select 2–5 products in total, then click **Compare**.
4. The comparison view is produced by P16: real product facts, side-by-side dimensions,
   MATCH/UNKNOWN/conflict relation to your active preferences, evidence sources, claim counts and
   the conditional recommendation text. No winner is invented when the evidence is tied or
   insufficient.
5. Try an invalid id (e.g. via a crafted API call): the BFF returns the real
   `invalid_product_reference` / `product_not_found` error, not a mock result.

### Scenario 3 — Clarify, update memory, continue

1. Start a **New conversation** with **Demo B · Memory write** (`profile_0057`).
2. Type: `I need some new headphones. Not sure what matters yet.`
3. The existing Decision Gate returns `NEED_USER_CLARIFICATION`
   (`insufficient_decision_criteria`); a clarification card shows the real question.
4. Type a real reply, e.g. `I prefer noise cancelling and wireless.` and submit.
5. The BFF calls the real MCP `submit_clarification_reply` with an idempotent turn id; M9 reports
   the real update (typically `added: 2`). The preference panel refreshes from
   `get_active_preferences`, and the pending query resumes through `recommend_products`.
6. Restarting the BFF resets the memory fixture (see §4).

### Other UI

- **Modes**: the header shows **CPU Live** as the live mode, **Qwen Replay** (clearly labelled
  *Historical Replay — Not a Live Model Call*) for saved trajectories, and **Qwen Live**
  disabled with its reason. Replay never calls a model.
- **Mobile**: width < 1280 becomes a single column; preferences, details/evidence and comparison
  open in drawers/sheets; the composer stays at the bottom.
- **Recovery**: if a tool fails, the UI shows the real error/status (with tool name and latency);
  it never paints a fake success. `/api/demo/restart` restarts both MCP servers if you need a
  clean memory state.
- **Recent conversations** are in-process; the browser remembers only the session id. BFF
  restart starts a fresh list.

## 6. Tests and verification

Run targeted tests only (the repository has a large unrelated test suite; do not run all of it).

Backend (WSL, repo root):

```bash
.venv/bin/python -m pytest -q tests/test_web_demo_unit.py
AGENTRECX_RUN_MCP_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_web_demo_integration.py
.venv/bin/python -m pytest -q tests/test_p17_mcp.py          # existing P17 contract still passes
```

Frontend (Windows):

```powershell
cmd /c "pnpm --dir web exec tsc --noEmit"
cmd /c "pnpm --dir web build"
```

End-to-end (real browser, real BFF/MCP; starts and stops the demo itself):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1
```

Latest verified result: **7/7 tests PASS**, 64 real evidence checks, 8 screenshots with validated
1440×900 / 390×844 sizes, ports released after cleanup. Artifacts:
`runs/web_demo/screenshots/`, `runs/web_demo/e2e/summary.json`,
`runs/web_demo/e2e/REPORT.md`, `runs/web_demo/e2e/evidence.jsonl`.

License/header audit:

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- .venv/bin/python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md
```

## 7. Troubleshooting

| Symptom | What to check |
|---|---|
| Page says "BFF unreachable" / readiness never ends | `runs/web_demo/logs/bff.console.err.log`; run the BFF manually: `wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- bash scripts/run_web_demo_bff.sh`. |
| `pnpm` cannot run in PowerShell | Use `cmd /c "pnpm ..."` (execution policy blocks the `.ps1` shim). |
| Port 3000/8787 already in use | The E2E launcher refuses with exit code 2; stop the other instance first (`scripts/stop_web_demo.ps1`). Do not blindly kill processes. |
| First search takes 20–35 s | Real BGE encoder warm-up on CPU; the tool timeline shows it and SSE keep-alives are sent. Subsequent searches are fast. |
| `memory_writes_disabled` on clarification reply | You are on Demo A (read-only). Create a conversation with Demo B, and start the BFF via the demo launcher (writes enabled). |
| BFF exits with `PermissionError` writing `runs/` | Run the BFF inside WSL as documented; do not use a Windows Python interpreter. |
| Memory changed unexpectedly / want a clean slate | Restart the BFF (fixture-backed M9); `POST /api/demo/restart` also restarts both MCP servers. |

## 8. Known limitations (honest)

1. **Catalogue evidence only.** No live price, stock, ratings, images, seller data or real-world
   quality can be verified; UNKNOWN is never filled.
2. **No product images.** The MCP catalogue payload exposes no image URL, so the UI uses a
   neutral self-drawn placeholder. No upstream product photos or brand assets are used.
3. **First-query latency** is CPU BGE warm-up (tens of seconds); this demo does not re-index,
   train or load another model to hide it.
4. **Memory is demo-local**: process memory, fixture-backed, reset on restart; there is no
   durable user database.
5. **No authentication / TLS / rate limiting**; localhost only.
6. **Qwen Agent mode is replay-only**; there is no live Qwen/GPU path in this demo.
7. Recommendation/comparison results are bounded catalogue results; they are not a claim of
   objective product quality or online shopping success.
8. The frontend is a local demo, not a production storefront: no cart, checkout, orders,
   accounts or payments.
