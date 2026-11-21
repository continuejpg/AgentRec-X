# AgentRec-X M3-B — Golden Dataset and Repeatable Agent Evaluation

> Status: implementation and a real 15-case CPU/BM25 evaluation run.  This is task-level
> evaluation only; no Before/After experiment, no Qwen, no algorithm change, no frontend.
> Quality remains **NOT ESTABLISHED**; engineering evaluation and recommendation quality are
> separate claims.

## 1. Design borrowed from mature tooling

The design follows the separation used by Phoenix Datasets/Experiments, DeepEval and MLflow
GenAI Evaluation:

- **Dataset and execution results are separate.** `agent_eval/golden/golden_agent_tasks_v1.json`
  contains expectations only; every run writes new `report.json`, `report.md` and
  `results.jsonl` under `runs/evaluations/<run_id>/`.
- **Expected outcome is explicit.** A case can expect a recommendation, a bounded refusal
  (`budget_unverifiable`/`budget_exceeded`), or a clarification. HTTP 200 alone is never a
  success.
- **The denominator is explicit.** Every metric reports numerator/denominator and only judged
  retrieval cases enter Precision@5.
- **Failure categories are explicit.** HTTP/execution failures, unexpected recommendations,
  missing refusals, wrong refusal codes, budget violations, replay failures and requirement
  mismatches are counted separately.
- **Reproducibility metadata is recorded.** Git SHA, dataset SHA256/version, catalogue/sidecar
  SHA256, BM25 configuration, price mode/FX, Python environment, command and timestamps are
  written into every report.

No Phoenix, MLflow, DeepEval or evaluation platform is installed into the project runtime.
The runner uses pytest-style Python code, the existing FastAPI/MCP/SQLite entry points and the
already implemented observability is left disabled (no Phoenix dependency).

## 2. Golden Dataset

**15 real tasks, dataset version `1.0.0`.** Each task has:

- stable `case_id`;
- user query and optional real clarification reply;
- expected structured requirements (`budget`, `category`, `use_cases`);
- expected business outcome;
- machine-checkable hard constraints;
- optional Harness-pinned acceptable product ids (pending human review);
- optional Harness-authored retrieval labels (pending human review);
- per-case dataset version and optional price mode.

| Category | Cases | What is evaluated |
|---|---:|---|
| normal | 2 | Normal/headphones and fully specified direct laptop recommendation |
| category_use_case | 3 | Gaming headphones, office laptop, tablet study |
| budget | 1 | Gaming headphones at a second budget value |
| budget_update | 1 (plus cold/replay executions) | 5000 → 8000 CNY budget revision |
| no_price | 1 | Strict-mode `budget_unverifiable` refusal |
| no_within_budget | 1 | `budget_exceeded` refusal |
| cold_resume | 3 | Recommendation across a real MCP child restart |
| replay | 3 | Completed reply replay with zero MCP |
| **total** | **15** | |

Cold Resume and Replay are explicitly separate cases, not inferred from an HTTP response.
The no-price case runs in **strict** price mode; all other cases run in
`demo_historical_fx`. The runner starts one BFF per price mode so the catalogue meaning cannot
silently leak between modes.

### Retrieval labels

Only three cases have pooled labels: `hp_noise_warm`, `laptop_update_warm` and
`laptop_direct_8000`. Every retrieved top-5 id was graded 0/1 by the Harness from the product
title/description; BM25 scores and ranks were not used as labels. **These labels are
machine-assisted and pending human review, not independent human verification.** `Precision@5 =
relevant / 5` over these fully judged pools. An unjudged id is excluded from the metric and
reported, never assumed non-relevant.

## 3. Real Runner

```bash
.venv/bin/python -m agent_eval \
  --dataset agent_eval/golden/golden_agent_tasks_v1.json \
  --output-dir runs/evaluations
```

The runner:

- starts the real FastAPI BFF and real BM25 MCP stdio child;
- uses isolated temporary Web State and M9 SQLite databases, never the developer DBs;
- executes real `POST /api/sessions`, `POST /api/chat/stream`,
  `POST /api/clarification/reply` and (for Cold Resume) `POST /api/demo/restart` calls;
- wraps the real `MCPManager.call_tool` only to record compact tool summaries; it does not
  replace or duplicate the Agent;
- measures end-to-end case wall time, excluding app/MCP startup;
- saves stable ids (`session_id`, `task_id`, `pending_id`, `reply_turn_id`,
  `recommendation_id`, HTTP request ids) for trace correlation; `trace_id` is `null` when
  tracing is disabled, as it is for this default runner;
- verifies replay equality and zero MCP calls for replay cases;
- verifies the MCP process identity actually changed for Cold Resume;
- runs with tracing disabled; Phoenix is not required.

## 4. Metrics

| Metric | Definition |
|---|---|
| Task Outcome Accuracy | task successes / all 15 cases; success requires the expected business outcome, correct refusal code or recommendation, and all hard constraints |
| Outcome-kind Accuracy | expected outcome category matches, without hard-constraint checks |
| Requirement Accuracy | passed `(budget, category, use_cases)` checks / 3 per case with observed structured requirements |
| Budget Violation Count / Rate | recommendations with unknown/over/foreign-currency price / all budget-constrained cases; correct refusals are not violations |
| Correct Refusals | HTTP 200 + expected refusal code + no product / expected refusal cases |
| Retrieval Precision@5 | relevant retrieved / 15 judged slots; only fully judged cases |
| End-to-End Latency | per-case wall time from session/ASK start to final reply or direct response; mean/p50/p95/max; app/MCP startup reported separately |
| Failure Types | execution HTTP errors, missing recommendations, unexpected recommendations, missing/wrong refusal, budget violation, replay and Cold Resume failures |

## 5. Recorded real run

The pre-commit run (`m3b-pre-02`, 15 cases) produced:

| Metric | Result |
|---|---|
| Task Outcome Accuracy | **13 / 15 = 86.7%** |
| Requirement Accuracy | **45 / 45 = 100.0%** |
| Budget Violations | **0 / 12 = 0.0%** |
| Correct Refusals | **2 / 2 = 100.0%** |
| Retrieval Precision@5 | **1 / 15 = 6.7%** (3 judged cases) |
| Execution failures | **2** (`laptop_office_6000`, `tablet_3000`; HTTP 503) |
| End-to-end latency | mean 8804.52 ms; p50 181.96 ms; p95 42219.35 ms; max 42240.13 ms |

The two failures are multi-round ASK limitations: the reply's merged requirements still lead
`recommend_products` to `awaiting_clarification`, which the durable coordinator reports as
HTTP 503 `clarification_reply_ambiguous`; no second ASK/pending exists in the current scope.
They are recorded as execution errors, not as correct refusals.

Cold Resume and Replay cases all passed. The no-price strict case returned a real
`budget_unverifiable` refusal. The no-within-budget case returned `budget_exceeded`. No
recommendation violated a budget; the retrieval metric exposes weak top-5 BM25 precision even
when the final recommendation is correct.

The final report generated after the milestone commit is authoritative for commit SHA,
dataset/catalogue hashes and exact latency. The canonical artifacts are:

- `runs/evaluations/<run_id>/report.json`
- `runs/evaluations/<run_id>/report.md`
- `runs/evaluations/<run_id>/results.jsonl`

The run is reproducible under the same code, dataset SHA256, catalogue/sidecar SHA256 and
configuration; latency naturally varies.

## 6. Explicit limits

- 15 tasks are a representative engineering fixture, not a statistically significant benchmark.
- The current CPU/BM25 Agent supports one ASK → one reply; office-laptop and tablet cases
  expose this limitation rather than being silently counted as successes.
- No nDCG: only binary relevance labels exist; no synthetic grades are invented.
- Demo FX is historical USD × fixed 7.00, not a live price.
- No Qwen, no GPU, no Before/After comparison in this milestone.
