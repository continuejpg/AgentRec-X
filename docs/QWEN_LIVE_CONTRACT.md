# Qwen Base Live — internal integration contract (P20)

Status: implementation contract for the Web Demo BFF, P17 MCP adapter and frontend.
Audience: backend/frontend owners in this round. This is not a public claim of live success.

## 0. Hard boundaries

- Qwen Base Live must call the **real Qwen3.5-9B Base `/plan` HTTP service** and feed its
  PlannerAction(s) through the **existing** `run_planner` + `ElectronicsPlannerExecutor`.
- No Rule→Qwen rewriting, no Qwen→Rule fallback, no fabricated PlannerAction, no bypass of
  product role / eligibility / CandidateLedger / Decision Gate / M9 / P15 evidence.
- Keep the existing 7 MCP tools' schemas and default behaviour. Add exactly one new tool:
  **`run_agent_turn`** (8 tools total), documented and tested as an addition.
- CPU Rule Live and Qwen Historical Replay remain available and unchanged in meaning.
- Qwen Live is **Unavailable** unless (a) explicitly enabled, (b) an endpoint is configured, and
  (c) a real health check returns a valid PlannerAction. No silent fallback.
- Never expose the model endpoint, credentials, or hidden chain-of-thought to the browser.
- Live integration may be `BLOCKED` when no service exists; that is recorded, not faked.

## 1. Environment configuration

BFF/runtime (not sent to browser):

| Variable | Meaning | Default |
|---|---|---|
| `QWEN_LIVE_ENABLED` | `1/true/yes/on` enables the optional mode | `0` |
| `QWEN_MODEL_ENDPOINT` | Base URL of the existing model server (e.g. a local tunnel URL) | empty |
| `QWEN_MODEL_NAME` | Informational model label only | `Qwen3.5-9B-Base` |
| `QWEN_LIVE_TIMEOUT_S` | Planner call timeout | `180` |
| `QWEN_LIVE_HEALTH_TIMEOUT_S` | Health-check timeout | `60` |
| `QWEN_LIVE_MAX_STEPS` | Bounded agent step cap (server also hard-caps) | `8` |
| `QWEN_LIVE_REQUIRE_HEALTH` | require valid PlannerAction health before availability | `1` |

Existing `AGENTRECX_QWEN_PLANNER` / `AGENTRECX_QWEN_PLANNER_URL` remain accepted as internal
aliases by the MCP server; the BFF maps its public config to them for the MCP child process.
The BFF must not start/stop AutoDL, create tunnels, download models, or touch GPU state.

## 2. Availability / health

`GET /api/health` adds (no endpoint/credential fields):

```json
"qwen_live": {
  "enabled": true,
  "configured": true,
  "status": "checking|available|unavailable|error|disabled",
  "available": false,
  "model": "Qwen3.5-9B-Base",
  "planner_action": null,
  "latency_ms": null,
  "checked_at": null,
  "detail": "No live Qwen service detected (disabled/unconfigured/unreachable/invalid PlannerAction)"
}
```

`GET /api/demo/config` modes:

```json
"qwen_live": {
  "available": false,
  "label": "Qwen Base Live",
  "detail": "Real Qwen3.5-9B Base PlannerAction → existing electronics executor → Gate/P15",
  "badge": "Live Qwen3.5-9B Base",
  "disabled_reason": "No live Qwen service detected …"
}
```

Rules:
- Default `disabled`/`unavailable`; CPU Live remains true; Replay remains true with its exact
  `Historical Replay — Not a Live Model Call` badge.
- If enabled+configured, BFF probes the real `/plan` endpoint in the background (valid action
  required) and may become available. A failed/late probe must not block BFF/MCP startup.
- On a later planner failure, the BFF marks Qwen unavailable (explicit error); it never silently
  runs Rule and labels it Qwen.
- Health calls are recorded with latency/action but are **not** counted as a user Qwen turn.

## 3. Chat API / SSE

`POST /api/chat/stream` request gains optional `"mode": "cpu_live" | "qwen_live"` (default
`cpu_live`; extra fields remain forbidden). Qwen Live request before availability → HTTP 409
`qwen_unavailable` (no SSE), with no fallback.

SSE additions (existing event names kept):
- `tool_status`: real progress from MCP progress notifications while `run_agent_turn` executes:
  - `qwen_planner` (label `Qwen Planner Request`) running/ok with model label + latency
  - `planner_action` (label `Planner Action: RETRIEVE|ASK|RERANK|FINALIZE`)
  - `retrieve`, `rerank`, `decision_gate`, `ask`, `finalize`, `evidence_preparation`
  Each row has real status/latency/detail from the execution trace. No hidden CoT.
- `done`: adds optional `"planner": "qwen_base"|"rule"`, `"planner_calls"`, `"planner_latency_ms"`.
- `products`/turn views: add optional planner metadata; product/evidence shapes otherwise unchanged.
- `error` codes: `qwen_unavailable`, `qwen_planner_timeout`, `qwen_planner_invalid`,
  `qwen_planner_error`. No fake FINALIZE.

Qwen clarification:
- If the real Qwen action is `ASK`, the MCP response carries a server-issued
  `clarification_id` and the model's question; BFF displays it and stores pending mode=qwen_live.
- On `POST /api/clarification/reply`, after the existing M9 `submit_clarification_reply` and
  preference refresh, the BFF resumes with `run_agent_turn` (not `recommend_products`) when the
  pending turn was Qwen. If Qwen ASK again, display honestly; no forced clarification.

## 4. P17 MCP addition: `run_agent_turn`

Input schema: `{"query": string}` (only). The server binds identity/search mode; the caller can
never pass user_id / memory flags / confirmed_by_user.

Output (structured content):

```json
{
  "schema": "agentrecx.p17.run_agent_turn.v1",
  "status": "finalized|awaiting_clarification|no_verified_candidate_in_scope|budget_exhausted|error",
  "query": "...",
  "planner": {
    "kind": "qwen_base",
    "model": "Qwen3.5-9B-Base",
    "calls": 2,
    "actions": [{"step": 1, "action": "RETRIEVE", "arguments": {...}, "rationale": "..."}],
    "latency_ms": [..],
    "input_tokens": 0, "output_tokens": 0,
    "parser_errors": []
  },
  "selected_product_id": "B0...|null",
  "candidate_ids": ["B0...", "..."],
  "gate": {...}|null,
  "evidence_bundle": {...}|null,
  "planner_steps": {...},
  "clarification": {"clarification_id": "...", "question": "...", "expires_in_seconds": 1800}|null,
  "search": {"search_mode": "bge|bm25", "score_kind": "...", "catalogue_size": 76263},
  "rag": {"final_answer": "...", "evidence_bundle": {...}, "claim_validation": {...}}|null,
  "error": null,
  "latency_ms": 1234.5
}
```

Failure: raise `ToolError("qwen_unavailable: ...")` / `qwen_planner_timeout` /
`qwen_planner_invalid` / `qwen_planner_error`; never return Rule output as Qwen.
Progress notifications use `ctx.report_progress(progress, 100, message_json)` where message is a
JSON object with `tool`, `label`, `status`, `detail`, `latency_ms` fields.

Planner/executor reuse:
- `HttpPlannerProposer(AGENTRECX_QWEN_PLANNER_URL)` from `recommendation.demo.planner_http_proposer`
- `run_planner(planner=..., executor=ElectronicsPlannerExecutor(...), max_steps<=8)`
- `HistoryStateReader` + `prepare_preferences/prepare_hard`
- `build_bundle` for P15; optional `EvidenceExplainer` only if `/explain` is configured/healthy.
- `run_agent_turn` must not write M9; writes remain only via `submit_clarification_reply`.

## 5. Tests and acceptance evidence

- Unit/contract: settings/modes/health; MCP tool schema 8 tools; Qwen unavailable path; fake
  planner HTTP server (clearly labelled test double, not Qwen) proving PlannerAction parsing,
  executor reuse, no Rule fallback, progress/trace; BFF SSE Qwen path with fake manager.
- Regression: existing Web E2E 7/7 unchanged in CPU/Replay modes; P17/P16/P15 targeted tests pass;
  CPU Rule Live default; Replay badge unchanged.
- Real live (only if a service is actually reachable): save real request/response action(s),
  model identity, tool observations, planner/explainer latencies, terminal/Gate, browser E2E
  screenshots. Otherwise `QWEN_LIVE_E2E = BLOCKED` in the report; no GO claim.

## 6. Report verdicts

`QWEN_MODEL_CONNECTION`, `QWEN_PLANNER_EXECUTION`, `MCP_EXECUTOR_INTEGRATION`,
`WEB_LIVE_E2E`, `MEMORY_CONTINUATION`, `CPU_RULE_REGRESSION`, `REPLAY_SEPARATION`;
overall `GO | LIMITED | BLOCKED | NO-GO`. `GO` requires real model inference and real browser
execution evidence. Code-only wiring cannot be GO.
