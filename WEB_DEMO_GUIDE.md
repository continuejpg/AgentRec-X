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
- **CPU Live vs Qwen Base Live vs Qwen Replay**: CPU Live is the default live mode. **Qwen Base
  Live is optional**: it becomes selectable only after the BFF's real `/plan` health check returns
  a valid PlannerAction from the configured Qwen3.5-9B Base service. Qwen Replay remains saved P18
  trajectories, always labelled **Historical Replay — Not a Live Model Call** and never presented
  as live. When no Qwen service is configured/reachable, Qwen Live shows **Unavailable** with the
  real reason; there is no silent Rule fallback.
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

- **Modes**: the header shows **CPU Live** as the default live mode, **Qwen Replay** (clearly
  labelled *Historical Replay — Not a Live Model Call*) for saved trajectories, and **Qwen Base
  Live** — selectable only when the real backend health check says available, otherwise disabled
  with its real reason. Replay never calls a model; Qwen Live never falls back to Rule.
- **Mobile**: width < 1280 becomes a single column; preferences, details/evidence and comparison
  open in drawers/sheets; the composer stays at the bottom.
- **Recovery**: if a tool fails, the UI shows the real error/status (with tool name and latency);
  it never paints a fake success. `/api/demo/restart` restarts both MCP servers if you need a
  clean memory state.
- **Recent conversations** are in-process; the browser remembers only the session id. BFF
  restart starts a fresh list.

## 6. Qwen Base Live (real GPU validated 2026-11-09)

This section now has **real GPU validation**, not just a contract wiring:
RTX 4090 24 GB, Qwen3.5-9B Base (`/root/autodl-tmp/models/Qwen3.5-9B-ms`, no SFT adapter), real
`/plan` calls, real MCP `run_agent_turn`, real executor/Gate/P15, and two real browser Qwen Live
turns. Full evidence: [`QWEN_LIVE_WEB_REPORT.md`](QWEN_LIVE_WEB_REPORT.md) and
`runs/web_demo/qwen_live/`.

This is an **optional** live mode over the existing P17 MCP / ElectronicsPlannerExecutor stack.
It is not a new Agent loop: the real Qwen3.5-9B Base returns a PlannerAction (`ASK`, `RETRIEVE`,
`RERANK`, `FINALIZE`), and the existing executor/search/scorer/ledger/eligibility/Gate/P15/M9
components execute it. A Rule→Qwen rewrite or Qwen→Rule fallback is not implemented.

### Configuration

Set these in the **same Windows PowerShell session** that launches the demo (the launcher forwards
them into WSL; the endpoint is never shown in the browser):

```powershell
$env:QWEN_LIVE_ENABLED = "1"                       # explicit opt-in
$env:QWEN_MODEL_ENDPOINT = "http://127.0.0.1:18009"  # existing model server, NOT auto-started
$env:QWEN_MODEL_NAME = "Qwen3.5-9B-Base"
$env:QWEN_LIVE_TIMEOUT_S = "180"                    # each /plan call
$env:QWEN_LIVE_HEALTH_TIMEOUT_S = "60"             # real PlannerAction health probe
$env:QWEN_LIVE_TURN_TIMEOUT_S = "600"              # bounds the complete multi-step turn
$env:QWEN_LIVE_MAX_STEPS = "8"
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
```

Availability is deliberately strict: the BFF performs a real `POST /plan` health probe and only
shows Qwen Live as available after it parses a valid PlannerAction. A configured-but-unreachable
service leaves **Qwen Live Unavailable**; CPU Live and Qwen Replay continue to work, and a Qwen
request never silently becomes a Rule request. The BFF does **not** start/stop AutoDL/GPU, create
SSH tunnels, download or quantize a model. Start the model service yourself, using the repository's
`training/agent_sft/model_server.py` or your existing endpoint.

The MCP server exposes the original 7 tools plus one conditional 8th tool,
`run_agent_turn`, only when Qwen Live is configured. `run_agent_turn` returns the model's real
actions, tool observations, Gate result, P15 evidence, planner/explainer latencies and the
optional Qwen clarification handle. `/api/chat/stream` accepts
`{"mode":"qwen_live", ...}` and emits the same SSE event names plus planner metadata on `done`.

### Last verified status (2026-11-09, real GPU)

- Instance: `autodl-container-06fa4ab99e-6a73b008`, GPU RTX 4090 24,564 MiB; CUDA true.
- Existing `training/agent_sft/model_server.py`, base model only (`adapter=None`), port 8009;
  model load **12.203 s**; GPU peak observed **19,845 MiB**; `/plan` 2.6–3.6 s direct/tunnel
  (2.6–4.7 s inside the agent loop); token counts 388–874 input / 69–107 output per call.
- Temporary WSL tunnel `127.0.0.1:18009 → remote 127.0.0.1:8009`; BFF health reported
  `qwen_live.status=available`, `planner_action=RETRIEVE`, both MCP servers exposed **8 tools**.
- Real MCP: 3 finalized turns (`B08D32JM5F`, `B07GPH9WF3`, `B00B93N0PO`, all
  `ALLOW_FINALIZE/evidence_sufficient`, P15 evidence/`rag` present) and one real non-FINALIZE
  (`under $1`: 7 bounded planner calls, Gate `NEED_MORE_PRODUCT_EVIDENCE`, no finalize).
- Real browser: full Playwright **9/9 PASS** (7 CPU/Replay + 2 real Qwen) **plus 1/1 separate
  real ASK→M9→resume browser test**; screenshots `qwen-live-real.png` / `qwen-live-nonfinal.png`
  and `runs/web_demo/qwen_ask_resume/browser_final/`.
- On 2026-11-10 the real Qwen ASK branch was triggered on an isolated conflict profile. A follow-up
  fix now treats "Please avoid wired headphones" as a **task-scoped hard exclusion**, overrides the
  old historical conflict only for this task, preserves the old M9 audit entry, re-evaluates the
  preserved candidate pool, and prevents an explicit-wired final product without requiring a second
  "Forget" reply. See [`QWEN_CLARIFICATION_FIX_REPORT.md`](QWEN_CLARIFICATION_FIX_REPORT.md); the
  final item's wireless evidence is UNKNOWN where the catalogue has no such evidence. No ASK or
  reply was fabricated.
- The temporary model server/tunnel were stopped after evidence collection; the AutoDL GPU
  instance lifecycle was not touched. Start the service and tunnel yourself for a rerun:

```bash
# remote GPU host: existing model_server.py + Base model (no adapter)
cd /root/autodl-tmp/agentrecx
nohup /root/autodl-tmp/agentrecx-sft/bin/python -m training.agent_sft.model_server \
  --model /root/autodl-tmp/models/Qwen3.5-9B-ms --port 8009 --max-new-tokens 256 \
  > /root/autodl-tmp/qwen_live_model_server.log 2>&1 &
```

```bash
# WSL: temporary tunnel (do not create/stop the GPU instance)
ssh -N -L 18009:127.0.0.1:8009 -p <port> -i <key> root@<host>
```

Then use the Qwen env block above and start `scripts/run_web_demo.ps1`. Qwen Live is selectable
only after `/api/health` reports `qwen_live.status=available`.

### Contract/regression tests (no GPU required)

```bash
# contract-only tests (a local HTTP planner double, clearly not Qwen)
.venv/bin/python -m pytest -q tests/test_qwen_live_unit.py tests/test_qwen_live_bff.py
AGENTRECX_RUN_QWEN_MCP_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_qwen_live_mcp_integration.py
```

For a real GPU rerun, the browser tests are
`e2e/tests/07-qwen-live-real.spec.ts` (requires a real available Qwen service).

## 7. Tests and verification

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

Latest verified result: **9/9 tests PASS** (7 original CPU/Replay + 2 real Qwen Live tests),
zero failures, real screenshots including `qwen-live-real.png` / `qwen-live-nonfinal.png`.
Artifacts:

- `runs/web_demo/qwen_live/e2e_real/summary.json`, `evidence.jsonl`, `full_suite_results.json`;
- `runs/web_demo/qwen_live/real_qwen_mcp_transcript.json`,
  `real_qwen_nonfinal.json`, `gpu_service_info.json`, `real_plan_*.json`;
- original CPU/Replay artifacts: `runs/web_demo/e2e/summary.json` and
  `runs/web_demo/screenshots/`.

License/header audit:

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- .venv/bin/python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md
```

## 8. Troubleshooting

| Symptom | What to check |
|---|---|
| Page says "BFF unreachable" / readiness never ends | `runs/web_demo/logs/bff.console.err.log`; run the BFF manually: `wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- bash scripts/run_web_demo_bff.sh`. |
| `pnpm` cannot run in PowerShell | Use `cmd /c "pnpm ..."` (execution policy blocks the `.ps1` shim). |
| Port 3000/8787 already in use | The E2E launcher refuses with exit code 2; stop the other instance first (`scripts/stop_web_demo.ps1`). Do not blindly kill processes. |
| First search takes 20–35 s | Real BGE encoder warm-up on CPU; the tool timeline shows it and SSE keep-alives are sent. Subsequent searches are fast. |
| `memory_writes_disabled` on clarification reply | You are on Demo A (read-only). Create a conversation with Demo B, and start the BFF via the demo launcher (writes enabled). |
| BFF exits with `PermissionError` writing `runs/` | Run the BFF inside WSL as documented; do not use a Windows Python interpreter. |
| Memory changed unexpectedly / want a clean slate | Restart the BFF (fixture-backed M9); `POST /api/demo/restart` also restarts both MCP servers. |
| Qwen Live stays Unavailable despite config | Check `/api/health` → `qwen_live` (`status`, `planner_action`, `latency_ms`, `detail`); verify `QWEN_LIVE_ENABLED=1`, `QWEN_MODEL_ENDPOINT` is reachable from the WSL BFF, and the service returns a valid `/plan` PlannerAction. The BFF never creates a tunnel or starts a GPU/model server. |
| Qwen Live request fails | The UI/API shows a real `qwen_*` error; it never falls back to Rule or Replay. Fix/reconnect the service, then retry. |

## 9. Known limitations (honest)

1. **Catalogue evidence only.** No live price, stock, ratings, images, seller data or real-world
   quality can be verified; UNKNOWN is never filled.
2. **No product images.** The MCP catalogue payload exposes no image URL, so the UI uses a
   neutral self-drawn placeholder. No upstream product photos or brand assets are used.
3. **First-query latency** is CPU BGE warm-up (tens of seconds); this demo does not re-index,
   train or load another model to hide it.
4. **Memory is demo-local**: process memory, fixture-backed, reset on restart; there is no
   durable user database.
5. **No authentication / TLS / rate limiting**; localhost only.
6. **Qwen Base Live is GPU-validated but not always-on**: real Qwen3.5-9B Base inference, real
   MCP/executor/Gate and two real browser turns passed on 2026-11-09. The temporary model service
   and tunnel are stopped; rerun requires starting them. On 2026-11-10 the real ASK→M9→resume branch
   was exercised and a follow-up fix made the current task constraint authoritative for the resumed
   candidate/Gate decision; see `QWEN_CLARIFICATION_FIX_REPORT.md` (GO for explicit-wired exclusion;
   wireless evidence remains UNKNOWN where the catalogue has none). Qwen Replay is still historical only.
7. Recommendation/comparison results are bounded catalogue results; they are not a claim of
   objective product quality or online shopping success.
8. The frontend is a local demo, not a production storefront: no cart, checkout, orders,
   accounts or payments.
