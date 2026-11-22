# AgentRec-X Web Demo — E2E acceptance (`e2e/`)

Independent Playwright project for the **real** local Web Demo. The suite drives the shipped
Next.js UI against the real FastAPI BFF, the real P17 MCP servers (BGE retrieval, P16 Gate /
comparison, P15 evidence, M9 memory) — **no mocks, no fabricated products/prices/preferences**.

The **CPU mandatory suite** is `01`–`06` plus the P1 demo `09`; `07`/`08` are the optional
**Qwen GPU** suite. Without a real Qwen service, `07`/`08` are reported as skipped, never as a
CPU failure. `09` is opt-in because it requires the two P1 flags and Demo FX mode.

- Browser: system Google Chrome via `channel: "chrome"` (no Playwright browser download).
- Trace: `retain-on-failure`; failure screenshot: `only-on-failure`; final acceptance screenshots
  are saved explicitly at exactly 1440×900 (desktop) / 390×844 (mobile).
- One worker, no retries (a retry could hide a real intermittent product bug).

## Run (recommended, one command)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1
```

The script: preflights ports 3000/8787 → installs this project's deps with
`PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` if needed → starts the demo with
`scripts/run_web_demo.ps1 -NoBrowser` → waits for real `GET /api/health` `ready=true` →
runs Playwright → stops the demo with `scripts/stop_web_demo.ps1` and verifies the ports were
released → writes `runs/web_demo/e2e/summary.json` + `REPORT.md`.

## Run the suite manually (demo already running and ready)

```powershell
cmd /c "cd /d D:\IT\CODE\PROJECTION\agentrec\X\e2e && pnpm exec playwright test"
node e2e/generate-report.mjs
```

CPU / Qwen split (demo already running):

```powershell
cmd /c "pnpm --dir e2e test:cpu"   # 01-06; 09 skips without P1 flags
cmd /c "pnpm --dir e2e test:qwen"  # 07/08; real Qwen GPU only
```

## What is verified

| Spec | Scenario | Real assertions |
|---|---|---|
| `tests/01-search-recommend.spec.ts` | Scenario 1: natural-language search / recommend (profile_0049) | real cards (ASIN ids/titles), fixed `Price unavailable / UNKNOWN`, UNKNOWN chips, `ALLOW_FINALIZE` Gate + selected candidate, P15 evidence raw field/span/source record/UNKNOWN, details panel |
| `tests/02-comparison.spec.ts` | Scenario 2: verified pair `B004K6R43G` vs `B00008VF63` | real POST `/api/products/compare` payload cross-checked against DOM (status/dimensions/UNKNOWN/claims/conditional/final answer), invalid refs → real 400/422 error |
| `tests/03-clarification-memory.spec.ts` | Scenario 3: vague query → clarification → reply → M9 write (profile_0057) | real clarification question, `memory_update.added=2`, `web_ui_explicit_user_reply`, refreshed wireless + noise cancelling, continued recommendation |
| `tests/04-mobile.spec.ts` | Mobile 390×844 | single column, left drawer, docked composer, real search, real comparison inside the sheet |
| `tests/05-modes.spec.ts` | Modes | CPU Live default real; `Historical Replay — Not a Live Model Call` visible; Qwen Live disabled with real reason; replay read-only |
| `tests/06-real-errors.spec.ts` | Real failure visibility | read-only vague request reaches `NEED_USER_CLARIFICATION`, never fabricates a Memory write or final recommendation; BFF 409/422 error boundaries are real |
| `tests/09-internship-demo.spec.ts` | P1 opt-in browser demo | laptop M4 ASK×2 + Demo FX budget, Memory Lab → formal candidate metadata, real Like → formal Behavior rerank metadata |

## Artifacts

All under `runs/web_demo/`:

- `screenshots/` — `desktop-search.png`, `desktop-evidence.png`, `desktop-comparison.png`,
  `desktop-clarification-memory.png`, `mobile-search.png`, `mobile-comparison-sheet.png`,
  `replay-mode.png` (+ `mobile-drawer.png`).
- `traces/` — Playwright traces; populated only for failed tests (`retain-on-failure`).
- `e2e/summary.json` — machine-readable result + launch/cleanup timings + evidence checks.
- `e2e/REPORT.md` — human acceptance report (real assertions, timings, screenshot/trace paths,
  known limitations).
- `e2e/evidence.jsonl` + `e2e/payloads/` — raw evidence records and cross-checked real HTTP payloads.

## Notes

- `e2e/` is self-contained; it is not added to `web/` and never modifies application code.
- The demo stack should be freshly started per acceptance run. In the default quick demo,
  Session/M9 state is process-local; the durable Web SQLite mode keeps Task/Pending/Request
  Ledger and is the mode that verifies MCP/BFF recovery and Replay.
- `07`/`08` require a real Qwen GPU service; absent it they skip. Do not report them as CPU
  passes, and do not delete them.
- Ports 3000/8787 must be free before `scripts/run_web_demo_e2e.ps1` starts; it refuses to kill
  processes owned by anyone else.
