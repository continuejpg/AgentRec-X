# AgentRec-X Setup and Deployment (V2.1)

> This document is the reproducible startup guide for the current Web Demo. It describes two
> verified modes:
>
> - **Mode A — Local CPU Demo (no GPU required)**
> - **Mode B — Qwen Live (remote AutoDL GPU + WSL-local tunnel)**
>
> The demo is local-only. No production multi-tenant deployment, auth or TLS is claimed.

## 0. What is verified

- CPU Live: real local BFF → MCP → BGE catalogue → Rule Planner / Gate / P15/P16 / M9.
- Qwen Live: real browser turn through BFF → MCP → Qwen3.5-9B → `RETRIEVE → RERANK → FINALIZE` →
  `ALLOW_FINALIZE / evidence_sufficient` (V2.1; 20.58 s single run, GPU 17,551 MiB).
- Session A/B M9 isolation; read-only writes refused with HTTP 400.
- Ranker Comparison is **recorded P100 Replay**, explicitly labelled.

Exact evidence: [`AGENTREC_X_DEMO_V21_VERIFICATION.md`](AGENTREC_X_DEMO_V21_VERIFICATION.md).

---

## 1. Repository and prerequisites

Expected repository root (examples):

```text
<REPO_ROOT>            # e.g. D:\...\AgentRec-X or /mnt/d/.../X
```

Verified development environment:

```text
Windows 10/11 + WSL2 Ubuntu-22.04
WSL Python 3.10 .venv at <REPO_ROOT>/.venv
Node.js + pnpm on Windows (web/package.json is the source of truth)
Chrome or Edge
```

The real catalogue/BGE assets are **not in Git**; they must exist in the author-provided local
workspace:

```text
data/p13r/electronics/catalog_meta.jsonl
data/enrichment/electronics_metadata_sidecar.jsonl
models/bge-base-en-v1.5
runs/p14d2/bge_index.npz
runs/p18/qwen_replay.json          # Qwen Replay
```

Check the frontend scripts in `web/package.json`; do not guess versions or ports:

```text
web/package.json
  dev       next dev -p 3000
  build     next build
  start     next start -p 3000
  typecheck tsc --noEmit
```

---

## 2. Mode A — Local CPU Demo (no GPU)

### 2.1 Install Python/BFF dependencies

From WSL, at repository root:

```bash
cd <REPO_ROOT>
.venv/bin/python -m pip install -r requirements.txt
```

If the `.venv` is missing, create it first (Python 3.10) and install the pinned requirements;
`requirements-mcp.txt` is the MCP-specific file referenced by older setup docs.

### 2.2 Install frontend dependencies

From Windows, at repository root:

```cmd
cd /d <REPO_ROOT>\web
pnpm install
```

### 2.3 Start BFF (WSL)

Recommended detached demo start (writes enabled, Demo B default):

```bash
cd <REPO_ROOT>
bash scripts/start_demo_bff_detached.sh
```

Or the regular one-command launcher from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
```

The BFF listens on:

```text
http://127.0.0.1:8787
```

Cold start is real: the two BGE/MCP servers load the catalogue + BGE index; in the verified host
this was ~134 s. The page renders before readiness.

### 2.4 Start frontend (Windows)

```cmd
cd /d <REPO_ROOT>\web
pnpm dev
```

Open:

```text
http://127.0.0.1:3000
```

Stop BFF:

```bash
kill "$(cat <REPO_ROOT>/runs/web_demo/bff.pid)"
```

### 2.5 What works without GPU

- CPU Live shopping chat.
- Real BGE catalogue search, product cards, UNKNOWN fields and P15 evidence.
- Rule Planner → Executor → Decision Gate.
- P16 product comparison.
- M9 preferences, Memory Timeline, Memory Lab (write-enabled profile).
- Session A/B isolation.
- Qwen Replay (historical only).
- P100 Ranker/Memory OFF–ON Replay panel.

Qwen Live is **not** available in Mode A. The UI shows Qwen Live Unavailable with the real reason;
there is no silent Rule fallback.

### 2.6 Replay in local mode

- Qwen Replay: `GET /api/replay` reads the saved historical trajectories; badge is
  `Historical Replay — Not a Live Model Call`.
- Ranker Replay: `GET /api/replay/rankers` reads the saved P100 GPU predictions; badge is
  `Recorded Experiment Replay — Not a Live Model Call`.

Neither endpoint starts a model or fabricates scores.

---

## 3. Mode B — Qwen Live (remote GPU + WSL-local tunnel)

### 3.1 Start AutoDL GPU instance

Start the existing AutoDL instance and confirm GPU availability:

```bash
nvidia-smi
```

Do not install/upgrade CUDA/PyTorch for this demo.

### 3.2 Start Qwen3.5-9B model service on AutoDL

Use the existing model and serving module; paths are environment-specific:

```bash
cd <AUTODL_REPO>
nohup <AUTODL_PYTHON> -m training.agent_sft.model_server \
  --model <AUTODL_QWEN_CHECKPOINT> \
  --port 18029 \
  --max-new-tokens 256 \
  > <AUTODL_LOG> 2>&1 &
```

Confirm it is listening and the GPU is used:

```bash
nvidia-smi
# service log should say: proposal server on 127.0.0.1:18029
```

### 3.3 Create a WSL-local SSH loopback tunnel

The working V2.1 setup uses a **WSL-side loopback tunnel**, not a Windows host-IP endpoint. This
matters because the BFF and the MCP child process both live in WSL and must use the **same
endpoint**.

From WSL:

```bash
ssh -N \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -i <PRIVATE_KEY_PATH> \
  -p <AUTODL_SSH_PORT> \
  -L 127.0.0.1:18029:127.0.0.1:18029 \
  <AUTODL_USER>@<AUTODL_HOST>
```

Keep this process running. Replace every `<...>` with your own values; no remote credentials are
stored in this repository.

Test the tunnel from WSL with a real `/plan` request (do not assume a port check is enough):

```bash
curl -s -X POST http://127.0.0.1:18029/plan \
  -H 'Content-Type: application/json' \
  -d '{"query":"wireless earbuds for working out","step":0,
       "allowed_actions":["RETRIEVE","ASK"],"candidate_count":0,"remaining_steps":8}'
```

A valid response contains a PlannerAction such as `RETRIEVE`, not just an HTTP socket response.

### 3.4 Start BFF with Qwen Live

In a new WSL shell, from repository root:

```bash
export QWEN_LIVE_ENABLED=1
export QWEN_MODEL_ENDPOINT=http://127.0.0.1:18029
export QWEN_MODEL_NAME=Qwen3.5-9B-Base
bash scripts/start_demo_bff_detached.sh
```

Both BFF and MCP child receive the endpoint through `MCPManager._server_parameters`; the endpoint
is never exposed to the browser.

Wait for:

```bash
curl -s http://127.0.0.1:8787/api/health
# modes.qwen_live.available == true
```

The BFF health monitor performs a **real** `POST /plan` and validates a PlannerAction; it is not
port-only.

### 3.5 Start frontend and verify

```cmd
cd /d <REPO_ROOT>\web
pnpm dev
```

Open `http://127.0.0.1:3000`, select **Qwen Live**, and send:

```text
I need wireless earbuds for working out.
```

Expected real trace:

```text
Qwen Planner Request (step 1) -> RETRIEVE
RETRIEVE -> 20 candidates
Qwen Planner Request (step 2) -> RERANK
RERANK -> candidates
Qwen Planner Request (step 3) -> FINALIZE
Decision Gate -> ALLOW_FINALIZE / evidence_sufficient
```

If Qwen is down, the UI must show Unavailable with a reason. It must not show a Rule result while
claiming Qwen.

### 3.6 Stop cleanly

- Stop tunnel: `kill <TUNNEL_PID>` in WSL.
- Stop BFF: `kill "$(cat <REPO_ROOT>/runs/web_demo/bff.pid)"`.
- Stop model service: `pkill -f training.agent_sft.model_server` on AutoDL.
- Confirm `nvidia-smi` returns to the expected idle state.

---

## 4. Environment variables (source-derived)

| Variable | Required | Purpose | Example |
|---|---|---|---|
| `AGENTRECX_WEB_DEMO_PORT` | No | BFF port | `8787` |
| `AGENTRECX_WEB_DEMO_SEARCH_MODE` | No | `bge` demo, `bm25` fast dev | `bge` |
| `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` | No | enable Demo B M9 writes | `1` |
| `AGENTRECX_WEB_DEMO_DEFAULT_PROFILE` | No | default session profile | `profile_0057` |
| `AGENTRECX_WEB_DEMO_MCP_TIMEOUT_S` | No | MCP tool timeout seconds | `180` |
| `AGENTRECX_WEB_DEMO_BGE_WARMUP` | No | BGE warmup at MCP startup | `1` |
| `AGENTRECX_WEB_DEMO_ASK_TEST_PROFILE` | No | expose isolated ASK profile | `1` |
| `AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG` | No | launcher owns BFF log | `1` |
| `QWEN_LIVE_ENABLED` | Mode B | opt into real Qwen Live | `1` |
| `QWEN_MODEL_ENDPOINT` | Mode B | real `/plan` endpoint | `http://127.0.0.1:18029` |
| `QWEN_MODEL_NAME` | No | display/model label | `Qwen3.5-9B-Base` |
| `QWEN_LIVE_TIMEOUT_S` | No | Planner request timeout | `180` |
| `QWEN_LIVE_HEALTH_TIMEOUT_S` | No | health probe timeout | `60` |
| `QWEN_LIVE_HEALTH_INTERVAL_S` | No | health interval | `60` |
| `QWEN_LIVE_TURN_TIMEOUT_S` | No | whole Qwen turn timeout | `600` |
| `QWEN_LIVE_MAX_STEPS` | No | planner step cap | `8` |
| `QWEN_LIVE_EXPLAIN` | No | enable explanation shadow | `1` |
| `NEXT_PUBLIC_BFF_URL` | No | browser BFF base | `http://127.0.0.1:8787` |

MCP child Planner env (`AGENTRECX_QWEN_PLANNER*`) is injected by the BFF into the trusted stdio
child; it is not a browser-facing setting.

---

## 5. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `/api/health` stays `starting` | BGE/MCP cold start | wait; cold start can be ~2 min in this workspace |
| `bff_unreachable` | BFF not running/port changed | start BFF; check `AGENTRECX_WEB_DEMO_PORT`/`NEXT_PUBLIC_BFF_URL` |
| Memory Lab disabled | read-only profile or writes off | switch to Demo B; set `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1` |
| Read-only write returns HTTP 400 | profile is read-only | switch to the write profile; expected behavior |
| Qwen Live Unavailable | no endpoint/health | check tunnel and model service; do not force a fallback |
| Qwen planner `planner_unavailable` | MCP child cannot reach endpoint | ensure BFF **and** MCP use the same `127.0.0.1:18029`; use a WSL-local loopback tunnel |
| Qwen health true but planner fails | health and child endpoint differ or tunnel path is flaky | compare health/child env; restart tunnel and BFF in order |
| Port 18029 occupied | another tunnel/model | stop the old tunnel/model; use an unused local port and update endpoint |
| Next.js port 3000 occupied | another dev server | stop it or run `next dev -p <PORT>` and use that URL |
| Session memory missing after restart | BFF process-local state | expected; restart loses session memory; no persistence claim |
| SSH tunnel drops | network/idle | restart tunnel, then BFF; verify `/api/health` and real `/plan` |
| GPU model stopped/OOM | remote model service | restart model; check `nvidia-smi`; demo should show Unavailable until healthy |
| Ranker panel unavailable | P100 replay artifact missing | check `runs/p100_extended_evaluation/p100_ranker_gpu_predictions.jsonl` |

---

## 6. Deployment limitations

- Verified: **local Windows/WSL demo + remote AutoDL Qwen Live**.
- Session Memory is **in-memory** in the BFF process; BFF restart loses it.
- No persistent multi-instance memory store has been verified.
- No production multi-tenant isolation/auth/TLS; the local demo binds loopback only.
- Qwen Live depends on a manually maintained SSH tunnel.
- Ranker Comparison is recorded Replay; not live inference.
- The catalogue has no trusted price/image/rating/stock; cards intentionally show UNKNOWN.
