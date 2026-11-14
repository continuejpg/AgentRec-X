# AgentRec-X Demo V2.1 — Verification

> Scope: only Qwen Live connectivity and browser-session memory isolation. No model training, no new
> Ranker, no new Memory Store, no production Scorer/Gate/Planner/P18 changes.

## 1. Qwen Live — PASS

### 1.1 Layered diagnosis

| Layer | Result |
|---|---|
| Qwen model server startup | PASS: `proposal server on 127.0.0.1:18029`; GPU 17,551 MiB |
| Server listen address | PASS: remote AutoDL `127.0.0.1:18029` |
| AutoDL-local `/plan` inference | PASS: HTTP 200, valid PlannerAction |
| WSL-local tunnel `/plan` | PASS after using WSL-side loopback tunnel; response 200 |
| BFF environment | PASS: `QWEN_LIVE_ENABLED=1`, endpoint `127.0.0.1:18029` |
| MCP child environment | PASS: child process env contained the same endpoint |
| BFF health check | PASS: real `/plan` + valid PlannerAction, not a port-only check |
| Actual browser Qwen turn | **PASS** |

### 1.2 Root cause

The earlier failure was a **transport/runtime configuration issue**, not a model or Planner semantic
bug:

- A Windows-side tunnel exposed through the Windows host IP was reachable by the BFF health path
  but the MCP child Planner request failed with `planner_unavailable`.
- The working configuration was a **WSL-side SSH loopback tunnel** at
  `127.0.0.1:18029`, with both the BFF and the MCP child using that same endpoint.
- The health monitor was already a real `POST /plan` + PlannerAction validation, not just a port
  probe. No fake fallback was added.

The only code change was preserving the real transport error cause in the Planner error text for
diagnosis; Planner semantics, action schema, allowed actions and fallback behavior are unchanged.

### 1.3 Actual browser Qwen Live result

Input:

```text
I need wireless earbuds for working out.
```

Mode: **Qwen Live** in the existing Next.js UI.

Raw first PlannerAction:

```json
{
  "action": "RETRIEVE",
  "arguments": {
    "query": "wireless earbuds for working out",
    "source_policy": "balanced"
  },
  "rationale": "The user has provided a clear intent ... retrieve candidates using the balanced source policy."
}
```

Full planner actions observed:

```text
RETRIEVE
RERANK
FINALIZE
```

Tool trace:

```text
Qwen Planner Request step 1 -> RETRIEVE
RETRIEVE -> 20 candidates
Qwen Planner Request step 2 -> RERANK
RERANK -> 1 candidate
Qwen Planner Request step 3 -> FINALIZE
Decision Gate -> ALLOW_FINALIZE (evidence_sufficient)
Evidence Preparation -> 0 evidence items
Qwen Agent -> ok
```

Final result:

- selected product: `B08D32JM5F`
- Gate: `ALLOW_FINALIZE` / `evidence_sufficient`
- planner calls: 3
- planner latency sum: 10,229.11 ms
- Qwen Agent E2E: 19,760.87 ms
- full turn wall: **20.58 s**
- GPU resident: 17,551 MiB

Records:

- [qwen_raw_planner_action.json](../runs/final_demo/qwen_raw_planner_action.json)
- [qwen_live_turn_v21.json](../runs/final_demo/qwen_live_turn_v21.json)
- [ui_qwen_live.png](../runs/final_demo/ui_qwen_live.png)

No silent Rule fallback is present; the Qwen call is visible in the UI with the Qwen Base Live badge.

## 2. Browser Session Memory Isolation — PASS

### 2.1 Minimal implementation

- Each browser session is passed as an opaque `session_id` to the memory-bearing MCP tools:
  `get_active_preferences`, `remember_preference`, `recommend_products`,
  `request_preference_clarification`, `submit_clarification_reply`, `run_agent_turn`,
  `compare_products`.
- `AppContext.memory_for(session_id)` creates/reuses an **existing**
  `ShoppingCompanionMemory` / `PreferenceMemoryService` per session. No new database, no copied
  memory rules.
- Read history uses the same existing `HistoryStateReader` with the session-scoped key.
- BFF passes its opaque session id; browser cannot inject a user id.
- No session-scoped executor is reused across turns; existing executor instantiation remains.

### 2.2 Actual two-session test

Records: [session_isolation.json](../runs/final_demo/session_isolation.json)

| Step | Result |
|---|---|
| A initial | active=[] |
| A writes `microphone` | added=1 |
| B reads | active=[] (does not see A) |
| B writes `over-ear` avoid | added=1 |
| A before withdrawal | microphone active |
| A withdrawal | removed=1 |
| A after | active=[]; microphone removed/inactive |
| B after A withdrawal | over-ear active unchanged |
| A recommendation | 10 products, `awaiting_clarification` |
| B recommendation | 10 products, `finalized` |
| Read-only profile write | HTTP 400 `memory_writes_disabled` |
| Reset A | HTTP 200; B memory unchanged |

This verifies:

- A cannot see B's memory;
- B cannot see A;
- B's withdrawal does not modify A;
- same-session write/read/withdraw works;
- reset does not touch other sessions;
- read-only profile refuses writes.

### 2.3 Persistence

Session-scoped M9 state is **in-memory for the BFF process**. Restarting the BFF loses all demo
session memory; this is shown in the UI notices and documented. No claim of durable demo memory.

## 3. Files changed

- `recommendation/mcp/server.py` — optional session-scoped M9 namespace; existing Writer/Reader.
- `recommendation/demo/planner_http_proposer.py` — preserves real transport error detail for
  diagnosis; no semantic change.
- `web_demo/demo_flow.py` — pass `session_id` through memory/recommend/clarification/comparison.
- `web_demo/app.py` — pass `session_id` for preferences/memory writes.
- `web_demo/config.py` — optional demo default profile and replay paths.
- `web_demo/viewmodels.py` — M9 timeline fields; P100 ranker replay view model.
- `web_demo/README.md` — new replay/memory endpoints and adapter note.
- `web/*` — Memory Timeline, Memory Lab, Ranker Replay Panel, scenario cards, API/types.
- `tests/test_web_demo_integration.py` — expected write-profile tool set includes the demo adapter.
- docs updated.

## 4. Tests

```text
38 passed  WSL: tests/test_web_demo_unit.py + tests/test_qwen_live_bff.py + tests/test_p17_mcp.py

pnpm exec tsc --noEmit  PASS
```

## 5. Known limitations

- Qwen Live requires a working local SSH tunnel from the WSL/BFF host to the AutoDL model server;
  the demo does not start tunnels or AutoDL automatically.
- Demo memory is process-local; BFF restart loses all session memory.
- Ranker Comparison remains recorded P100 Replay, explicitly labelled; not live inference.
- Browser QA was headless Edge screenshot + real HTTP/SSE; not a multi-user auth deployment.

## 6. Final startup

CPU Live:

```bash
bash scripts/run_web_demo_bff.sh
```

Qwen Live (requires an already-running local tunnel at `127.0.0.1:18029`):

```bash
export QWEN_LIVE_ENABLED=1
export QWEN_MODEL_ENDPOINT=http://127.0.0.1:18029
bash scripts/run_web_demo_bff.sh
```

Frontend:

```bash
cd web && pnpm dev
```

Open `http://127.0.0.1:3000`.

## 7. Position

V2.1 closes the two V2 blockers:

- **Qwen Live is actually reachable through Browser → BFF → MCP → Qwen → PlannerAction → Executor → Gate.**
- **Browser sessions have independent M9 memory namespaces** while reusing the existing event-sourced
  Writer/Reader.

The production default remains Existing Scorer; no model or ranker decision changes in this round.
