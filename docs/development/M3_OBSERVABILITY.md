# AgentRec-X M3-A Observability

> Status: implementation + real local Phoenix acceptance at the M3-A commit.
> Scope is tracing only. No Golden Dataset, evaluator, Qwen, or recommendation-algorithm
> changes are part of this milestone.

## 1. What was adopted

| Concern | Adopted open-source component | Why |
|---|---|---|
| Trace/span lifecycle, W3C context, batching | Official OpenTelemetry Python SDK `1.45.2` | Standard API; no custom trace IDs, contextvars or exporters |
| HTTP server spans | Official `opentelemetry-instrumentation-fastapi` `0.66b2` (ASGI) | Records method/route/status/duration without reading bodies |
| OTLP export | Official `opentelemetry-exporter-otlp-proto-http` `1.45.2` | Phoenix receives standard OTLP/HTTP at `/v1/traces`; bounded, batched export |
| MCP client/server spans | MCP Python SDK `2.3.0` built-in server `OpenTelemetryMiddleware`, plus one thin client span per real call | Reuses the official server boundary; the client span measures the real stdio call and carries the W3C carrier. No second MCP client or proxy was added |
| Cross-process context | W3C `traceparent`/`tracestate` in MCP request `_meta` | MCP 2.3.0 forwards `_meta`; its server middleware extracts it. Real parent/child link, not a fabricated one |
| Phoenix span kinds | OpenInference semantic conventions (`openinference.span.kind`) | Makes Agent/Tool/Retriever/Chain stages legible in Phoenix |
| Phoenix UI/backend | Arize Phoenix `20.16.0`, isolated server environment | Local OTLP receiver and trace UI; no custom trace database or UI |

We did not add Langfuse/MLflow/Opik as dependencies, and did not introduce an Agent framework.
Phoenix is a receiver/UI only; the OpenTelemetry SDK remains the instrumentation authority.

The implementation uses the standard OTLP/HTTP exporter directly rather than
`phoenix.otel.register()`. `register()` is convenient for a global auto-instrumented app, but
it mutates the global provider and hides the disabled/no-op path that this project needs for
tests and for the "tracing off" contract. The Phoenix OTLP endpoint and protocol are unchanged.

### Version note

Phoenix 20.17+ requires Python 3.11+, while this project's Web Demo virtualenv is Python
3.10. The accepted setup therefore uses **two environments**:

* `.venv` (Python 3.10) runs the BFF, MCP child and OTel client instrumentation;
* an isolated Python 3.12 environment runs `arize-phoenix==20.16.0`.

No CUDA/PyTorch packages are touched, and Phoenix is not installed into the project runtime
environment.

## 2. Local setup

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X

# Client instrumentation in the existing project environment.
.venv/bin/python -m pip install -r requirements-observability.txt

# Isolated Phoenix server environment (one-time; outside the repo).
uv venv /tmp/agentrecx-phoenix-venv --python 3.12
uv pip install --python /tmp/agentrecx-phoenix-venv/bin/python 'arize-phoenix==20.16.0'

# Start Phoenix.  Its SQLite file is under the gitignored runs/ tree.
PHOENIX_HOST=127.0.0.1 PHOENIX_PORT=6006 \
PHOENIX_TELEMETRY_ENABLED=false \
PHOENIX_SQL_DATABASE_URL=sqlite:////mnt/d/IT/CODE/PROJECTION/agentrec/X/runs/observability/phoenix.sqlite3 \
/tmp/agentrecx-phoenix-venv/bin/phoenix serve
```

Start the BFF with tracing explicitly enabled:

```bash
AGENTRECX_OTEL_ENABLED=1 \
AGENTRECX_OTEL_ENDPOINT=http://127.0.0.1:6006 \
AGENTRECX_OTEL_PROJECT=agentrecx-x \
.venv/bin/python -m web_demo
```

Tracing is **disabled by default**. Without `AGENTRECX_OTEL_ENABLED=1` the observability
module does not create a provider or exporter, and business behavior is unchanged.

## 3. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AGENTRECX_OTEL_ENABLED` | `0` | Explicit opt-in; no implicit tracing |
| `AGENTRECX_OTEL_ENDPOINT` | `http://127.0.0.1:6006/v1/traces` | OTLP/HTTP endpoint; a base URL gets `/v1/traces` appended |
| `PHOENIX_COLLECTOR_ENDPOINT` | unset | Fallback endpoint, compatible with Phoenix docs |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Standard OTLP fallback |
| `AGENTRECX_OTEL_PROJECT` | `agentrecx-x` | `openinference.project.name` resource attribute |
| `AGENTRECX_OTEL_SERVICE_NAME` | `agentrecx-web-demo` | BFF service name; child uses `agentrecx-mcp` |
| `AGENTRECX_OTEL_SAMPLE_RATIO` | `1.0` | Parent-based trace-id ratio sampler |
| `AGENTRECX_OTEL_EXPORT_TIMEOUT_S` | `2.0` | OTLP request and batch export timeout |

Exporter settings are passed to each MCP child through its environment. The endpoint is not
exposed by `/api/health` or `/api/demo/config`; health exposes only enabled/project/service/
exporter/error.

## 4. Trace shape

A Cold Resume reply has this real hierarchy (one trace; stdio child spans share the same
trace ID through W3C propagation):

```text
POST /api/clarification/reply                  SERVER   FastAPI/ASGI
└── agent.reply                              AGENT    warm/cold/replay/checkpoint_finalize
    ├── reply.handle_check                    CHAIN
    │   └── resume.cold_restore               CHAIN
    │       └── mcp.call restore_clarification_handle   TOOL (BFF client)
    │           └── tools/call restore_clarification_handle  SERVER (MCP child)
    ├── reply.claim                           CHAIN
    ├── reply.external_started                CHAIN
    ├── reply.external_execution              CHAIN
    │   ├── mcp.call submit_clarification_reply      TOOL -> MCP SERVER
    │   ├── mcp.call get_active_preferences          TOOL -> MCP SERVER
    │   ├── mcp.call recommend_products              TOOL -> MCP SERVER
    │   │   ├── retrieval.catalog_search       RETRIEVER
    │   │   ├── budget.assessment              CHAIN
    │   │   └── gate.decision                  CHAIN
    │   └── mcp.call get_product_info                TOOL -> MCP SERVER
    ├── reply.checkpoint                      CHAIN
    └── reply.finalize                        CHAIN
```

`mcp.call` is the BFF client boundary and `tools/call` is the real child server boundary; they
are two measurements of one invocation, not a second tool execution. No business tool is
called twice for tracing.

ASK creates `agent.chat` under `POST /api/chat/stream`, with the same MCP child hierarchy.
Replay creates `agent.reply` with `agentrecx.reply.path=replay`; it has no tool-call children.
A `RESULT_DURABLE` restart creates `agent.reply` with
`agentrecx.reply.path=checkpoint_finalize` and no MCP call.

### Stable correlation attributes

* `agentrecx.session_id` — Web session (safe random id; not a message)
* `agentrecx.task_id` — Web task
* `agentrecx.pending_id` — clarification pending
* `agentrecx.reply_turn_id` / `agentrecx.request_id` — ledger idempotency identity
* `agentrecx.clarification_id` — real MCP handle id (old and new handle ids differ after
  Cold Resume; the stable pending/task ids remain)
* `mcp.process_instance_id` / `mcp.generation` — which real child served a tool call
* `agentrecx.http_request_id` on HTTP spans — per-request correlation id, also in
  `X-AgentRecX-Request-Id` and the BFF log line (distinct from the ledger request id below)
* `agentrecx.request_id` on reply/ledger spans — the stable reply-turn id used by the Request
  Ledger
* `agentrecx.reply.path` — `warm`, `cold_resume`, `replay`, `checkpoint_finalize` or a failure
  attempt
* `agentrecx.execution_stage` — `CLAIMED`, `EXTERNAL_STARTED`, `RESULT_DURABLE`, `COMPLETED`

The ASK and post-restart reply are separate HTTP traces but share `agentrecx.session_id`,
`agentrecx.task_id` and `agentrecx.pending_id`; this is the lookup key across processes.

Phoenix reserves `session.id` for its UUID-based session feature. Web session ids are not
UUIDs, so they are carried as `agentrecx.session_id` instead of forcing a misleading session
grouping.

## 5. Privacy and safety

* No user message, query text, preference value, product title, narrative, tool argument
  values or tool response bodies are recorded.
* Tool spans record the tool name and sorted **argument keys** only; product/price fields are
  represented by counts, status codes, currency and selected ids.
* Exception messages are not exported. Only `error.type` / `mcp.error.code` are attached; the
  real exception still propagates.
* HTTP instrumentation captures route/status and a random request id. Request/response headers
  and bodies are not captured. `/api/health` polling is excluded to avoid noise.
* The SQLite Request Ledger remains authoritative. Tracing never decides claim, retry,
  checkpoint, finalize or replay.
* Exporter failure is contained by the OpenTelemetry batch processor and by the observability
  facade; a failed collector cannot turn a successful recommendation into an error or trigger
  a second MCP call.
* `Tracing.shutdown()` flushes with a bounded timeout. Child shutdown flushes its own provider
  after the stdio server returns. A hard SIGKILL can lose the last unexported batch; that is
  an observability gap, not a business-state gap.

## 6. Verification

```bash
# Focused tracing regression (no Phoenix required; in-memory exporter).
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_observability_tracing.py -q -p no:cacheprovider

# Real Phoenix + real BM25 MCP + real Cold Resume + real demo FX + replay.
AGENTRECX_RUN_OBSERVABILITY_E2E=1 \
AGENTRECX_PHOENIX_URL=http://127.0.0.1:6006 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_web_observability_e2e.py -q -p no:cacheprovider
```

The opt-in E2E asserts the actual Phoenix `/v1/projects/<project>/spans` response contains
the Span names above, the `mcp.call` client → `tools/call` server parent link, the budget
counts (`within=1`, `unknown=3`, fixed `7.00` demo FX), the cold-resume path, and that replay
has no MCP tool children. It also checks that raw query/reply/product text is absent from
every exported attribute.

M3-A recorded results (this commit):

* `tests/test_observability_tracing.py`: **9 passed**;
* real Phoenix E2E: **1 passed in 77.51s**;
* full relevant Web/M1/M2 suite:
  **295 passed, 2 skipped in 926.32s**, exit code 0 (the two skips are the opt-in
  `test_web_demo_integration` and the opt-in Phoenix E2E, which was run separately above);
* real degraded run with `AGENTRECX_OTEL_ENDPOINT=http://127.0.0.1:9`: ASK → restart →
  Cold Resume still returned HTTP 200 with the MSI `B0BY2Z27LC` result and `within=1`.
  The collector failures were logged by the SDK and did not change the business result.

## 7. References used

* OpenTelemetry Python: <https://opentelemetry.io/docs/languages/python/>
* OpenTelemetry FastAPI/ASGI instrumentation:
  <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html>
* OTLP HTTP exporter configuration:
  <https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/>
* MCP Python SDK's server OpenTelemetry middleware:
  <https://github.com/modelcontextprotocol/python-sdk> (`mcp/server/_otel.py` in `2.3.0`)
* Phoenix local terminal setup: <https://arize.com/docs/phoenix/self-hosting/deployment-options/terminal>
* OpenInference semantic conventions: <https://github.com/Arize-ai/openinference>

The project does not use an LLM-provider OpenInference auto-instrumentor because the accepted
CPU/BM25 path has no LLM SDK call. The generic Agent/Tool/Retriever spans are created explicitly
with OpenInference span kinds.

## 8. Known limits

* Observability is local-only; no collector authentication, TLS, remote deployment or
  multi-process queue is configured.
* No metrics or structured-log export is included. Request logs carry request/trace ids, but
  this milestone is tracing-only.
* A hard-killed MCP child may lose its last batch; graceful restart flushes.
* The E2E uses the real catalogue and sidecar and therefore takes about 80 seconds in this
  environment (two BM25 child startups dominate).
* No dual-BFF/distributed trace testing, no arbitrary crash-point tracing matrix, no Golden
  Dataset or evaluation claim.
