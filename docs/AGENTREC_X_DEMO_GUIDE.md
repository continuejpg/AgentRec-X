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

### Behavior Inspector (Behavior Feedback V1.1)

Open the header **Behavior** button or right panel → **Behavior**.  The panel shows:

- real browser events: impressions (visible card ≥60% for 1000 ms), clicks, compare
  attempt/completed/failed and short-term Like/Dislike;
- the first-party short-term profile: verified attribute interests with product/evidence refs,
  source event types and the 30-minute decay half-life;
- the difference between **explicit M9 memory** (long-term, user-confirmed writes only) and
  **short-term behavior**;
- **Eligible OFF / ON** only for Gate `ALLOW_FINALIZE` candidates in
  `candidate_presentable_ids`, plus a separate, clearly-labelled **Diagnostic Ranking — Not Final
  Recommendations** for every displayed candidate;
- a behavior-inferred preference suggestion that is **never written automatically**; the existing
  M9 writer runs only after **Remember this preference**, and the button is disabled on the
  read-only Demo A profile;
- a **Personalization ON/OFF** switch and **Delete session events**.

A quick demo:

```text
A. Search "I need wireless headphones for the gym.", scroll cards only:
   impressions appear, but short-term interests stay at zero.
B. Click a card and Like it; search again:
   click/Like produce weak interests and the next Eligible Behavior Shadow may move a rank.
C. Search "I need wireless earbuds for the gym." and compare two cards:
   Compare failed is recorded separately from Compare attempted; no success is fabricated.
D. Open Behavior on that earbuds recommendation:
   the Gate has one eligible candidate; ineligible search cards appear only in the diagnostic list.
E. Start a new conversation: zero click/Like events from the previous session.
F. Turn Personalization OFF: no behavior contribution, normal recommendations continue,
   and M9 explicit preferences are untouched.
```

The formal shadow is **eligible-only**: it uses the same query/candidates/metadata/positions/
eligibility, Active M9 memory and Gate; ineligible products cannot enter the eligible Top-K.  The
production product list above the panel is not re-ranked.  Only `LIVE_USER_INTERACTION` enters the
profile; Replay/synthetic events are tagged and excluded.  Impressions are exposure context only;
compare attempts/completed/failed are operation provenance, not preferences.  Event storage is a
local SQLite file under `runs/web_demo/behavior/`; no external analytics SDK is used.  See
[`AGENTREC_X_BEHAVIOR_FEEDBACK_V11_REPORT.md`](AGENTREC_X_BEHAVIOR_FEEDBACK_V11_REPORT.md),
[`AGENTREC_X_BEHAVIOR_FEEDBACK_V1_REPORT.md`](AGENTREC_X_BEHAVIOR_FEEDBACK_V1_REPORT.md) and
[`BEHAVIOR_FEEDBACK_TECH_SELECTION.md`](BEHAVIOR_FEEDBACK_TECH_SELECTION.md).

The V1.1 package also includes a real-data blind-review package under
`runs/behavior_feedback_v11/` (`blind_review_tool.html`, `human_review_tasks.jsonl`,
`human_review_template.csv`).  No human labels were auto-generated; without real labels the result
file reports `preference_lift = unavailable_no_human_labels`.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `AGENTRECX_WEB_DEMO_SEARCH_MODE` | `bge` for demo, `bm25` for fast dev only | `bge` |
| `AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES` | enables writes for `profile_0057` | `1` in launcher |
| `AGENTRECX_WEB_DEMO_PORT` | BFF port | `8787` |
| `QWEN_LIVE_ENABLED` / `QWEN_MODEL_ENDPOINT` | optional real Qwen Live; never silently faked | disabled |
| `AGENTRECX_WEB_DEMO_ASK_TEST_PROFILE` | optional isolated ASK test profile | disabled |
| `AGENTRECX_BEHAVIOR_ENABLED` | master switch for first-party behavior tracking | `1` |
| `AGENTRECX_BEHAVIOR_DB` | SQLite path for behavior events | `runs/web_demo/behavior/events.sqlite3` |
| `AGENTRECX_BEHAVIOR_WINDOW_S` | short-term profile window | `21600` (6 h) |
| `AGENTRECX_BEHAVIOR_HALF_LIFE_S` | decay half-life | `1800` (30 min) |
| `AGENTRECX_BEHAVIOR_RETENTION_DAYS` | event retention / cleanup | `7` |

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
- Behavior Feedback V1 records only local impressions/clicks/compares/Like-Dislike; it has no
  purchase/conversion event, no propensity log, and no CTR/IPS/quality-lift claim.  The Behavior
  shadow is experimental and never the production default.

## Common failures

| Symptom | Cause | Action |
|---|---|---|
| Composer disabled, `/api/health` `starting` | cold BGE/MCP startup | wait; watch readiness banner |
| `bff_unreachable` | BFF not running/port changed | start BFF, check `NEXT_PUBLIC_BFF_URL` |
| Memory Lab disabled | read-only profile or writes env off | switch to write-enabled profile; restart with writes=1 |
| Qwen Live unavailable | no Qwen endpoint/health | use CPU Live/Replay; do not force a fallback |
| Rankers unavailable | P100 replay artifact missing | check `runs/p100_extended_evaluation/` |
| `clear session` does not clear memory | by design | M9 is user/profile state; restart BFF to reset local fixture |
