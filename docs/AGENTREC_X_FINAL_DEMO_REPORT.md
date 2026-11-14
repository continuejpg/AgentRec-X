# AgentRec-X Final Demo V2 — Verification Report

> 本轮为现有 AgentRec-X 构建展示层，没有训练新模型、没有替换生产 Scorer/Gate/Planner/P18、没有新建 Memory Store 或第二套 Agent。
> 本报告只记录实际运行结果，不把 Replay 说成 Live。

## 1. 最终选择

**复用现有 `web/` Next.js 16 + `web_demo/` FastAPI BFF。**

没有引入 Chainlit/Gradio/React 重写：
- Chainlit 已官方归档，不维护；
- 现有 Next.js UI 已具备聊天、SSE、商品卡、Evidence、Comparison、Trace、Replay；
- 新 Demo 的主要工作是补齐 Memory Timeline、Memory Lab、P100 Ranker Replay 和三个 Scenario 入口。

详见 [DEMO_TECH_SELECTION.md](DEMO_TECH_SELECTION.md)。V2.1 的 Qwen Live 与 Session Isolation 验收见 [AGENTREC_X_DEMO_V21_VERIFICATION.md](AGENTREC_X_DEMO_V21_VERIFICATION.md)。 与 [AGENTREC_X_DEMO_ARCHITECTURE.md](AGENTREC_X_DEMO_ARCHITECTURE.md)。

## 2. 实际运行路径

```text
Next.js UI :3000
→ FastAPI BFF :8787
→ MCP stdio servers (profile_0049 read-only / profile_0057 write-enabled)
→ existing recommendation.mcp.server
→ BGE catalogue search / existing Planner+Executor / M9 Memory / Gate / P15 / P16
```

本轮实际启动：
- BFF: `http://127.0.0.1:8787`，`ready=true`；
- UI: `http://127.0.0.1:3000`，HTTP 200；
- BFF cold start measured: ~134 s for the two BGE MCP servers;
- SQLite-free in-process M9 demo memory, reset on BFF restart.

## 3. Three actual scenarios

Recorded file: [walkthrough_three_scenarios.json](../runs/final_demo/walkthrough_three_scenarios.json).

### A — Remember my preference

1. Query: `I need some new headphones. Not sure what matters yet.`
2. Real `search_products` → `get_product_info` → `recommend_products` → real ASK.
3. Reply: `I prefer a microphone.`
4. Real M9 update: `added = 1`, active preference `microphone`.
5. `Recommend again using my updated preferences.` → `finalized`.
6. Product card path is real BGE catalogue output; no hand-selected product.

Recorded:
```text
clarification reason: missing_key_need
M9 source: web_ui_explicit_user_reply
active: microphone (prefer, free_form_constraint, logical_seq 1)
```

### B — Negative / update preference

1. Add `I do not want over-ear headphones.` through the explicit M9 Memory Lab adapter.
2. Real M9 update: `added = 1`, `over-ear headphones` as avoid.
3. `Recommend headphones using my current preferences.` → `finalized`.
4. The recommendation is recomputed by the real Agent; no frontend reranking.

### C — Withdraw a preference

1. The existing microphone preference is active.
2. `Forget my microphone preference.` through the same M9 writer.
3. Real M9 result: `removed = [microphone_memory_id]`, `removal_directives = 1`.
4. Reader active state: microphone no longer active.
5. Timeline shows the original event with status **WITHDRAWN**; history is not deleted.
6. `Recommend again after my preference withdrawal.` → `finalized`.

Product IDs and full tool rows are in the walkthrough JSON. They are produced by the real Agent path; this report does not claim a specific Top-1 must be “correct”.

## 4. Memory lifecycle verification

Observed in the live BFF:

- Active preferences panel updates after real M9 writes.
- Memory timeline rows include:
  - memory id;
  - preference value;
  - kind/polarity;
  - logical sequence;
  - source turn;
  - status (`active`, `removed`, `superseded`);
  - source text.
- Withdrawal is shown as a state change, not deletion.
- The Memory Lab writes through the existing `PreferenceMemoryService.process_turn`; no alternate store.
- The recommendation is only recomputed when a subsequent chat turn is sent; the memory write itself does not fake ranking changes.

Screenshot: [ui_memory_lab.png](../runs/final_demo/ui_memory_lab.png)

## 5. Shopping UX / UI QA

Actual Edge headless screenshots:
- [ui_home_v2.png](../runs/final_demo/ui_home_v2.png) — home, scenario cards, Active Memory, timeline, default write demo profile;
- [ui_memory_lab.png](../runs/final_demo/ui_memory_lab.png) — timeline + Memory Lab + real prefs;
- [ui_rankers_final.png](../runs/final_demo/ui_rankers_final.png) — Rankers tab with explicit Replay badge and Memory OFF/ON recorded ranks.

Checked:
- chat input and SSE rendering;
- real tool rows;
- ASK card;
- product cards don't fabricate price/stock/rating/images;
- Active Memory + timeline;
- ranker replay panel;
- read-only/write profile distinction;
- Qwen Live real turn evidence and no silent fallback;
- empty and readiness states.

`pnpm exec tsc --noEmit` passes for the demo frontend.

## 6. P100 Ranker / Memory OFF–ON Replay

The right panel **Rankers** reads:
`runs/p100_extended_evaluation/p100_ranker_gpu_predictions.jsonl`.

It shows:
- Existing Scorer;
- ShopRanker Natural (Experimental / Shadow);
- Fixed Hybrid (Experimental / Shadow);
- recorded Memory OFF/ON top-5, rank movement, ranker score, evidence count, memory matches/conflicts/UNKNOWN.

It always displays:

```text
Recorded Experiment Replay — Not a Live Model Call
```

No live ShopRanker/Hybrid inference is fabricated. Recorded Memory OFF/ON is not a live memory toggle.

## 7. Live GPU vs Replay — V2.1

### Qwen Live — PASS

A real browser Qwen Live turn completed:

```text
Browser → BFF → MCP → Qwen3.5-9B
→ RETRIEVE → RERANK → FINALIZE
→ ALLOW_FINALIZE (evidence_sufficient)
→ selected B08D32JM5F
```

- planner calls: 3; planner latency sum: 10,229.11 ms;
- Qwen Agent E2E: 19,760.87 ms; full turn 20.58 s;
- GPU resident: 17,551 MiB;
- endpoint: WSL-local `127.0.0.1:18029` via an existing tunnel;
- no silent Rule fallback, no faked action.

Root cause of the earlier failure: the Windows host-IP tunnel was reachable by the BFF health path
but the MCP child Planner request failed. Switching BFF and MCP child to the WSL-local loopback
tunnel fixed the real PlannerAction path. The health monitor already performs a real
`POST /plan` + PlannerAction validation.

Records: [qwen_live_turn_v21.json](../runs/final_demo/qwen_live_turn_v21.json),
[qwen_raw_planner_action.json](../runs/final_demo/qwen_raw_planner_action.json),
[ui_qwen_live.png](../runs/final_demo/ui_qwen_live.png).

### CPU Live (actual)

Real BFF → real MCP → real BGE/rule planner/Gate/M9/P15; scenario A/B/C ran through this path.
This remains the stable default.

### Qwen Replay

Existing read-only historical Qwen trajectories remain available and labeled historical.

### Ranker Replay

P100 ranker/Memory OFF–ON remains recorded replay; it does not claim live ShopRanker/Hybrid inference.

## 8. Framework reuse QA

### Reused directly

- `web/` Next.js App Router, React 19, Tailwind, SSE client, chat UI, product cards, evidence panel, comparison panel, tool timeline, session UI, replay UI, readiness UI.
- `web_demo/` FastAPI BFF, session store, MCP manager, viewmodels, health/config.
- Existing MCP tools for search/info/recommend/preferences/clarification/compare.
- Existing M9 writer/reader, existing Agent/Gate/Scorer, existing P100 predictions.

### Thin custom additions

- `web/components/MemoryTimeline.tsx` — M9 audit timeline (existing fields).
- `web/components/MemoryLab.tsx` — explicit M9 write UI.
- `web/components/RankerReplayPanel.tsx` — recorded P100 ranker replay.
- scenario cards in the existing empty-chat state.
- BFF `/api/replay/rankers` and `/api/memory/remember`; one thin MCP `remember_preference` adapter.

No duplicate chat framework, no duplicate memory DB, no duplicate ranking implementation.

## 9. Known limitations — V2.1

1. **Session isolation is implemented for the local demo.** Each browser session gets its own
   existing `ShoppingCompanionMemory` namespace; A/B tests pass. The state is still process-local:
   restarting the BFF loses all session memory. It is not a durable multi-tenant production store.
2. **Qwen Live requires a running local tunnel** to the AutoDL model server. The demo does not
   start AutoDL or tunnels automatically.
3. Memory Lab is an explicit demo-only write adapter; normal user preference writes still go through
   the real clarification flow.
4. The ranker panel is recorded P100 replay, not live ShopRanker/Hybrid inference.
5. Browser QA was headless screenshot/HTTP/SSE based, not a full multi-user auth deployment.
6. Catalogue has no trusted price/image/rating/stock fields; cards correctly show UNKNOWN/placeholder.

## 10. Start commands

BFF (WSL), from repo root:

```bash
bash scripts/run_web_demo_bff.sh
```

Detached variant used in this session:

```bash
bash scripts/start_demo_bff_detached.sh
```

Frontend (Windows):

```cmd
cd /d D:\IT\CODE\PROJECTION\agentrec\X\web
pnpm dev
```

Open `http://127.0.0.1:3000`.

## 11. Demo verification summary

| Item | Status |
|---|---|
| Real BGE search / product cards | PASS |
| Real clarification + M9 write | PASS (scenario A) |
| Real negative preference | PASS (scenario B) |
| Real withdrawal + history retained | PASS (scenario C) |
| Active Memory updates | PASS |
| Memory Timeline | PASS |
| Real Agent trace / tool rows | PASS |
| Ranker / Memory OFF–ON replay | PASS (recorded, labeled) |
| Qwen Live | PASS (real browser turn) |
| Qwen Replay | available (historical) |
| Session Isolation | PASS (A/B write/withdraw/reset; read-only write refused) |
| Fake price/rating/image | none |
| Production Scorer/Gate/Planner/P18 modified | no |
| New model training | no |
| Git Push | no |

## 12. Final position

The demo is suitable for a local interview walkthrough as:

> “A real shopping Agent with event-sourced long-term memory, real catalogue evidence, real Gate decisions, and an honest separation between live execution and recorded ranker experiments.”

It is **not** claimed to be multi-tenant production software: Qwen Live depends on a local tunnel,
session memory is process-local, and Ranker Comparison remains recorded replay.
