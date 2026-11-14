# AgentRec-X Final Demo V2 — Demo Guide

## Quick start

From the repository root, in WSL Ubuntu-22.04:

```bash
bash scripts/run_web_demo_bff.sh
```

This starts the FastAPI BFF on `127.0.0.1:8787` with:
- `AGENTRECX_WEB_DEMO_SEARCH_MODE=bge`;
- `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1`;
- two long-lived MCP stdio servers for the two fixed demo profiles.

The detached demo script used in this session additionally sets
`AGENTRECX_WEB_DEMO_DEFAULT_PROFILE=profile_0057` so the page opens on the write-enabled
**Demo B · Memory write** profile. The normal one-command launcher still defaults to Demo A;
select Demo B in the sidebar before trying a live memory write.

Cold start is real: the MCP servers load the catalogue + BGE index (~2 minutes on the current
host). The page renders immediately and shows the real starting state; the composer unlocks when
`/api/health` reports `ready: true`.

On Windows PowerShell, a detached start is used in this session:

```bash
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/d/IT/CODE/PROJECTION/agentrec/X && bash scripts/start_demo_bff_detached.sh'
```

Start the existing Next.js UI (second terminal):

```cmd
cd /d D:\IT\CODE\PROJECTION\agentrec\X\web
pnpm dev
```

Open <http://127.0.0.1:3000>.

Stop the BFF:

```bash
kill "$(cat runs/web_demo/bff.pid)"
```

## What the demo shows

### Shopping chat + products

Each turn uses the real existing pipeline:
`search_products → get_product_info → recommend_products → Gate → P15 evidence`.
No product is invented, no price/rating/stock is fabricated, and UNKNOWN remains UNKNOWN.

### Active Memory + Memory timeline

The sidebar shows:
- current active positive/negative preferences;
- M9 event history with `logical_seq`, status, source turn, polarity and supersession/withdrawal IDs;
- real removed/inactive history, not a deleted row.

### Memory Lab

On the write-enabled profile, explicit user text can be sent through the real M9 writer:
- `I prefer a microphone.`
- `I do not want over-ear headphones.`
- `Forget my microphone preference.`

This uses the same M9 `PreferenceMemoryService` and audit trail as a clarification reply. It does
not rerank by itself; say **“Recommend again using my updated preferences.”** to run a new real
recommendation turn.

### Scenario cards

The empty chat shows three real prompts:
- **A — Remember my preference**
- **B — Update / negative preference**
- **C — Withdraw a preference**

Recommended walkthrough:

```text
A: "I need some new headphones. Not sure what matters yet."
   → clarification
   → reply "I prefer a microphone."
   → "Recommend again using my updated preferences."

B: "I prefer a microphone."
   → "Recommend again using my updated preferences."
   → Memory Lab: "avoid over-ear" / "I do not want over-ear headphones."
   → "Recommend again using my updated preferences."
   → inspect Active Memory / timeline.

C: "I prefer a microphone."
   → "Recommend again using my updated preferences."
   → Memory Lab: "withdraw microphone" / "Forget my microphone preference."
   → inspect the removed event in the timeline
   → "Recommend again using my updated preferences."
```

The recommendation product IDs/ranks are produced by the real catalogue/Agent path; they are
not manually arranged.

### Rankers / Memory OFF–ON

Right panel → **Rankers** loads recorded Phase 5.1 P100 GPU predictions:
- Existing Scorer;
- ShopRanker Natural (Experimental / Shadow);
- Fixed Hybrid (Experimental / Shadow).

It shows recorded Memory OFF/ON top-5, rank movement, matches/conflicts/UNKNOWN and the explicit
badge **Recorded Experiment Replay — Not a Live Model Call**. It never reranks or calls a model.

### Qwen Replay / Live

Qwen Replay is historical and read-only. Qwen Live is available only when the BFF's real
`POST /plan` health probe returns a valid PlannerAction.

To use Qwen Live with the existing AutoDL server, start the model server, keep a **WSL-local**
loopback tunnel on `127.0.0.1:18029`, then start the BFF with:

```bash
export QWEN_LIVE_ENABLED=1
export QWEN_MODEL_ENDPOINT=http://127.0.0.1:18029
bash scripts/run_web_demo_bff.sh
```

The UI then exposes **Qwen Live** with a live badge. If the tunnel/model is unavailable the UI
shows Unavailable and never silently falls back to Rule.

Verified in V2.1: Browser → BFF → MCP → Qwen → RETRIEVE/RERANK/FINALIZE → Gate, E2E 20.58s,
GPU 17,551 MiB.

### Session Memory Isolation

Each browser session is passed as an opaque `session_id` to the existing M9 tools. The MCP server
creates a separate existing `ShoppingCompanionMemory` namespace per session. No new store or rules
were added. Session A/B cannot read or withdraw each other's preferences; read-only profile writes
return HTTP 400.

Memory is process-local and resets when the BFF restarts; the UI and test evidence say this
explicitly. Reset Session clears turns only and never another session's memory.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `AGENTRECX_WEB_DEMO_SEARCH_MODE` | `bge` for demo, `bm25` for fast dev only | `bge` |
| `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` | enables writes for `profile_0057` | `1` in launcher |
| `AGENTRECX_WEB_DEMO_PORT` | BFF port | `8787` |
| `QWEN_LIVE_ENABLED` / `QWEN_MODEL_ENDPOINT` | optional real Qwen Live; never silently faked | disabled |
| `AGENTRECX_WEB_DEMO_ASK_TEST_PROFILE` | optional isolated ASK test profile | disabled |

## Known limitations

- BGE/catalogue cold start is ~2 minutes.
- HTTP browser interaction is local-only; no public deployment/auth.
- M9 memory writes are process-local and reset when the BFF restarts.
- Session memory is isolated per browser session through the existing M9 service, but is
  process-local and resets when the BFF restarts.
- The ranker panel is recorded P100 replay; no live ShopRanker/Hybrid inference is claimed.
- Qwen Live requires the local tunnel/model service; it is verified in V2.1 but not auto-started.
- Product images/prices/ratings/stock are absent/UNKNOWN because the catalogue has no trusted
  source for those fields.

## Common failures

| Symptom | Cause | Action |
|---|---|---|
| Composer disabled, `/api/health` `starting` | cold BGE/MCP startup | wait; watch readiness banner |
| `bff_unreachable` | BFF not running/port changed | start BFF, check `NEXT_PUBLIC_BFF_URL` |
| Memory Lab disabled | read-only profile or writes env off | switch to write-enabled profile; restart with writes=1 |
| Qwen Live unavailable | no Qwen endpoint/health | use CPU Live/Replay; do not force a fallback |
| Rankers unavailable | P100 replay artifact missing | check `runs/p100_extended_evaluation/` |
| `clear session` does not clear memory | by design | M9 is user/profile state; restart BFF to reset local fixture |
