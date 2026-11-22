# AgentRec-X M3 Engineering Baseline

> **Status: M3-A/B/C verified at the M3-D archive point. Engineering evidence is recorded;
> recommendation quality remains NOT ESTABLISHED.** This page distinguishes verified
> engineering behavior from exploratory retrieval evidence and from still-unreviewed labels.
>
> M3-D did not add an algorithm, did not integrate Category-scoped BM25, and did not enlarge
> the Golden Dataset.

## 1. M3-A — Observability

### Adopted open-source components

| Concern | Component |
|---|---|
| Trace/span lifecycle, W3C context, batching | OpenTelemetry Python SDK `1.45.2` |
| FastAPI/ASGI server spans | `opentelemetry-instrumentation-fastapi==0.66b2` |
| OTLP/HTTP export | `opentelemetry-exporter-otlp-proto-http==1.45.2` |
| MCP server spans | MCP Python SDK `2.3.0` server `OpenTelemetryMiddleware` |
| Span kinds | OpenInference semantic conventions `0.1.46` |
| Local trace UI/collector | Arize Phoenix `20.16.0` in an isolated Python 3.12 environment |

Phoenix 20.16 is used because newer Phoenix releases require Python 3.11+ while the Web
virtualenv is Python 3.10. The standard OTLP/HTTP exporter is configured directly instead of
`phoenix.otel.register()` so tracing has an explicit disabled path and per-process providers.

### Trace mechanism

- BFF client spans (`mcp.call <tool>`) inject W3C `traceparent`/`tracestate` into MCP request
  `_meta`.
- MCP 2.3.0 server middleware extracts that carrier and emits `tools/call <tool>` as a real
  child span in the stdio child process.
- The child also emits `retrieval.catalog_search`, `budget.assessment` and `gate.decision`.
- The BFF records `agent.chat`, `agent.reply`, `reply.handle_check`, `resume.cold_restore`,
  `reply.claim`, `reply.external_started`, `reply.external_execution`, `reply.checkpoint`,
  `reply.finalize` and replay paths.
- Stable keys: `agentrecx.session_id`, `task_id`, `pending_id`, `reply_turn_id`,
  `request_id`, `mcp.process_instance_id`, `mcp.generation`, `agentrecx.reply.path`.
- Tracing is disabled by default and does not depend on Phoenix for business execution.

Verified files: `observability/`, `tests/test_observability_tracing.py`,
`tests/test_web_observability_e2e.py`, `web_demo/mcp_manager.py`, `web_demo/reply_executor.py`,
`recommendation/mcp/server.py`.

## 2. M3-B — Golden Dataset and task evaluation

### Dataset

- **15 tasks**, version `1.0.0`.
- File: `agent_eval/golden/golden_agent_tasks_v1.json`.
- Categories: normal (2), category/use case (3), budget (1), budget update (1), no price (1),
  no within-budget (1), Cold Resume (3), Replay (3).
- Each case has stable id, query, optional reply, expected requirements, expected outcome,
  hard constraints, optional acceptable product ids, optional pooled labels, dataset version
  and price mode.
- The no-price case uses **strict** price mode; other cases use `demo_historical_fx` with the
  fixed `7.00` demo rate. The runner starts one BFF per mode.

### Runner

`python -m agent_eval` starts the real FastAPI BFF and real BM25 MCP stdio children against
temporary Web State/M9 SQLite databases. It drives real `/api/sessions`,
`/api/chat/stream`, `/api/clarification/reply` and `/api/demo/restart` calls, records compact
tool summaries, verifies Replay equality/zero MCP, and checks Cold Resume process identity.
Tracing is disabled; Phoenix is **not** required.

### Verified task results

The corrected verification run after M3-D is recorded under `runs/evaluations/m3b-verify-01/`.
Metrics:

| Metric | Result | Denominator |
|---|---:|---|
| Task Outcome Accuracy | **13 / 15 = 86.7%** | 15 tasks |
| Requirement Accuracy | **45 / 45 = 100%** | 3 fields × 15 observed-requirement cases |
| Budget Violations | **0 / 12 = 0%** | 12 budget-constrained cases |
| Correct Refusals | **2 / 2 = 100%** | 2 expected-refusal cases |
| Raw-query Precision@5 | **1 / 15 = 6.7%** | 3 fully judged cases × 5 |
| Execution failures | **2** | 15 tasks |

Failure cases:

- `laptop_office_6000`: HTTP 503 `clarification_reply_ambiguous`.
- `tablet_3000`: HTTP 503 `clarification_reply_ambiguous`.

Both are caused by the M3-era **single ASK → one reply** boundary: after merging the reply,
`recommend_products` still returned `awaiting_clarification`, and the durable coordinator
reported 503 rather than inventing a second pending or retrying M9. They are execution
failures, not correct refusals.

> **M4 note:** the CPU/BM25 multi-round boundary was subsequently implemented. If the M3-B
> suite is re-run on M4 code, those two cases expose a real next ASK instead of the 503. The
> numbers above remain the historical M3 archive at the M3 commit.

Important metric boundaries:

- M3-B raw-query Precision@5 measures the BFF **display search** raw query. M3-C Baseline
  measures the current production **recommendation effective query**. They use different
  query sets and denominators and must never be combined.
- M3-B labels are Harness-authored binary labels, **pending human review**, not independent
  human verification.
- BM25 actual parameters: `k1=1.2`, `b=0.75`; `search_products` top_k=10. The earlier M3-B
  report writer hard-coded `k1=1.5`; that metadata bug was corrected and the verification run
  above uses the real constant.
- `M3B` result is a fixture result, not a statistical benchmark.

## 3. M3-C — Retrieval experiment

### What was compared

- **Baseline:** current production recommendation effective query
  `augment_query(raw, extract_task_requirements(raw))`, full catalogue BM25.
- **Candidate:** the same query and BM25 parameters, but a category-scoped index. The only
  experiment variable is retrieval scope.
- 10 queries, top-5 per strategy, 50 judged slots per strategy.
- Pooled annotation audit: **90 query-product rows, 79 unique product ids, 40 positive
  labels, 21 repeated product appearances across queries, 0 unjudged top-5 rows**. The 21
  repeated appearances are the same product retrieved by multiple queries, not duplicate
  annotation rows.
- Metric recomputation from the stored rows produced exactly 11/50 and 37/50; no row counts
  or P@5 arithmetic discrepancies were found.
- Production `recommend_products` retrieves `TOPK=20`; the experiment truncates to top-5 for
  P@5 evaluation. The top-5 ordering is identical to the first five of a 20-result BM25 list;
  the experiment's latency is specifically **top-5 search latency**, not full production
  top-20 latency.

### Verified fixture metrics (`m3c-final-01`, final clean run)

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Precision@5 | **11 / 50 = 22.0%** | **37 / 50 = 74.0%** | **+52.0 pp** |
| Error/accessory share | 39 / 50 = 78.0% | 13 / 50 = 26.0% | -52.0 pp |
| Search-only median sum | 419.18 ms | 6.97 ms | — |

Per-query: all 10 improved on this fixture, with no observed degradation. Laptop gaming top-5
changed from keyboard skins/covers/headset to real gaming laptops; general laptop top-5 changed
from stands to real ASUS/Lenovo/MSI laptops. Office and study queries reached only 1/5 because
the use-case match remains weak.

The experiment is reproducible: `m3c-final-02` produced the same per-query relevant counts and
the same 11/50 vs 37/50 aggregate.

### Scope audit — why no production integration

A diagnostic audit (`runs/m3_audit/scope_audit.md`) found material issues:

- Laptop scope contains 457 members; 7,258 of 7,620 title-type diagnostic products are excluded.
  Real laptops with "Hard Drive"/"Keyboard" in the title, and headphones bundled with a carrying
  case, are among obvious false exclusions.
- Headphone scope 1,993; 1,527 title-type products excluded; real products with "Case", "CD" or
  "Adapter" can be excluded.
- Mouse scope 377; a real Microsoft wireless mouse can be excluded by "Battery"/"Base".
- Keyboard scope 559; 1,787 title-type products excluded (many genuine accessories, but the
  exclusion rule is lexical).
- Tablet scope 339 is sidecar-only; it contains false positives such as a mini PC and
  "Silvia Osteo Plus Tablets".

Consequences: the high P@5 is **exploratory category purity**, with no recall estimate and no
human qrels. The candidate is not integrated. `recommend_products`, `search_products`, Budget
Gate, requirement merging, Finalize and Replay remain unchanged.

## 4. Labels and review status

| Artifact | Provenance | Status |
|---|---|---|
| M3-B Golden labels | Harness-authored from titles/descriptions | `pending_human_review` |
| M3-C query labels | Harness-authored from titles/descriptions/use cases | `pending_human_review` |
| Human review | Not performed | **not verified** |

M3-C provides a review sheet with empty human columns:
`runs/retrieval_experiments/m3c-final-01/annotation_review.csv` (and `.md`). The machine labels
and reasons are preserved, but the status must not be changed to verified without an independent
human reviewer.

No Recall/MRR/nDCG is reported for M3-C because the judged pool is only the top-5 union, not a
complete corpus qrels set. No Recall is constructed from incomplete labels.

## 5. Environment and reproducibility

| Item | Value |
|---|---|
| Repository | `D:\IT\CODE\PROJECTION\agentrec\X` / WSL `/mnt/d/IT/CODE/PROJECTION/agentrec/X` |
| Python | 3.10.12 (Web venv); Phoenix uses isolated 3.12 |
| Compute | CPU only; no GPU, no Qwen |
| Catalogue | accepted C4 Electronics `catalog_meta.jsonl`, 76,263 records |
| Catalogue SHA256 | `14f27eedd94915ad3c5372ab3a8e13cd13f432240fd572db37e4681c49aaa1e6` |
| Sidecar SHA256 | `2b76df5b650b5d017dcb9918d871e6584b793de6c05d7dac97e9df6aef1bda07` |
| BM25 | `k1=1.2`, `b=0.75`, mode `bm25` |
| Price mode | strict or demo_historical_fx; FX fixed `7.00` |
| M3-B dataset version | `1.0.0` (current SHA recorded in verification report) |
| M3-C query dataset | `1.0.0`, SHA256 `324f5e785ef32dc48c260650aa0205d02c520c6b1f9a91707b7dd1cb0603d980` |

Reproduce:

```bash
# M3-A tests; real Phoenix E2E requires a local Phoenix at :6006
.venv/bin/python -m pytest tests/test_observability_tracing.py -q
AGENTRECX_RUN_OBSERVABILITY_E2E=1 AGENTRECX_PHOENIX_URL=http://127.0.0.1:6006 \
  .venv/bin/python -m pytest tests/test_web_observability_e2e.py -q

# M3-B real task suite (no Phoenix)
.venv/bin/python -m agent_eval --output-dir runs/evaluations

# M3-C retrieval experiment
.venv/bin/python -m agent_eval.retrieval --output-dir runs/retrieval_experiments
```

## 6. M3-D verification record

Actually run in this round:

| Check | Result |
|---|---|
| M3-A observability unit | **9 passed** |
| M3-A real Phoenix E2E | **1 passed in 80.65s** |
| M3-B/C + M3-A unit set | **30 passed** (includes 6 M3-C, 15 M3-B, 9 M3-A) |
| M2 key regression | **52 passed in 169.29s** |
| M1 demo-FX smoke | **1 passed in 101.68s** |
| M3-C final report | `m3c-final-01`, 11/50 vs 37/50, exit 0 |
| M3-C repeat | `m3c-final-02`, identical per-query counts, exit 0 |
| M3-B corrected verification | `m3b-verify-01`, expected 13/15 and real K1=1.2 |
| Historical M1/M2/M3-A wide suite | 295 passed, 2 skipped on the M3-A commit; **not rerun in M3-D** |

No production recommendation code was modified in M3-D. The only code/metadata changes in the
M3-C line were the experiment package and a factual BM25 `k1` metadata correction; M3-D itself
adds documentation and annotation-provenance wording only.

## 7. Remaining technical limits

1. **No independent human relevance review**; M3-B and M3-C labels remain machine-assisted.
2. **Category scope has recall/coverage issues and sidecar false positives**; not production.
3. **No full-corpus recall evaluation**; only pooled top-5 labels.
4. **M3-era single ASK → one reply**: office/tablet tasks failed with 503 at the M3 archive;
   M4 subsequently implements the CPU/BM25 durable next ASK.  This M3 record is historical.
5. **No integrated candidate run**: no M3-B downstream suite exists for a category-scoped
   production path.
6. **Small samples**: 15 tasks and 10 retrieval queries are engineering fixtures; no
   statistical-significance claim.
7. **M3-A is local observability**, no collector auth/TLS, no metrics/log pipeline.
8. **Historical 295-test result** is not a rerun of the full repository; the 261-file repo has
   not been fully executed in M3-D.

Conclusion: M3-A and M3-B are closed as engineering evidence; M3-C is a verified,
reproducible **negative-on-production-readiness**/positive-on-fixture experiment. It proves the
category-scope idea is worth further investigation, not that it is safe to ship.
