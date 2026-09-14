# AgentRec-X — Usage

Copy-paste instructions for running the demo, the API, the tests and the smokes.

Every command in this document was executed against the current repository state. Where a
command is expensive or has not been safely re-run, it is labelled as such.

Cross-references: [README](../README.md) · [Architecture](ARCHITECTURE.md) ·
[Experiments](EXPERIMENTS.md).

---

## Environment assumptions

* The project virtualenv already exists at `.venv/`. It was created with
  `--system-site-packages` so it reuses the system PyTorch/CUDA stack; **do not reinstall
  or upgrade CUDA/PyTorch**.
* Python 3.10.8; the accepted benchmark environment is PyTorch 2.1.2+cu118 on an RTX 4090,
  but everything in this document runs on **CPU**.
* All commands are relative to the repository root (`/root/AgentRec-X`).
* The demo deliberately imports no provider SDK and needs no network access and no API key.

```bash
cd /root/AgentRec-X
.venv/bin/python -m pip check      # expected: "No broken requirements found."
```

---

## Required artifacts

These files are **intentionally not committed** (`.gitignore` excludes `data/`, `runs/`,
`*.pt`, `*.jsonl`, `*.db`, `*.sqlite3`). The application discovers them by fixed
repository-relative paths, overridable with environment variables.

| Artifact | Default path | Used by |
| --- | --- | --- |
| Accepted SASRec checkpoint | `runs/sasrec_canonical_2026/best.pt` | inference, demo |
| Run manifest | `runs/sasrec_canonical_2026/run.json` | checkpoint cross-check |
| Item mappings | `data/processed/Sports_and_Outdoors_mappings.json` | inference |
| Processed sequences | `data/processed/Sports_and_Outdoors_sequences.json` | demo profiles |
| Normalized catalogue metadata | `data/processed/Sports_and_Outdoors_products.jsonl` | M8 enrichment |
| Preference-memory SQLite | `data/artifacts/demo/preference_memory.sqlite3` | demo (created on demand) |
| Raw review data | `data/raw/Sports_and_Outdoors.jsonl.gz` | preprocessing only |
| Raw product metadata | `data/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz` | metadata build only |

Verified digests of the accepted artifacts:

```text
352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912  runs/sasrec_canonical_2026/best.pt
e3549049f955bb0540444c221523444b2a333e8df5eac6b97984342e40c6e2c7  runs/sasrec_canonical_2026/run.json
175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd  data/processed/Sports_and_Outdoors_products.jsonl
```

Verify them yourself:

```bash
sha256sum runs/sasrec_canonical_2026/best.pt \
          runs/sasrec_canonical_2026/run.json \
          data/processed/Sports_and_Outdoors_products.jsonl
```

The server verifies the checkpoint digest at startup when
`AGENTRECX_VERIFY_CHECKPOINT` is enabled (default), and it refuses to start if a required
artifact is missing.

---

## Environment variables

All variables are optional. Defaults match the accepted repository layout.

| Name | Purpose | Default | Required | Example |
| --- | --- | --- | --- | --- |
| `AGENTRECX_CHECKPOINT_PATH` | SASRec checkpoint | `runs/sasrec_canonical_2026/best.pt` | no | `/models/other.pt` |
| `AGENTRECX_MANIFEST_PATH` | run manifest for cross-checks | `runs/sasrec_canonical_2026/run.json` | no | `/models/run.json` |
| `AGENTRECX_MAPPINGS_PATH` | `parent_asin ↔ item_id` mappings | `data/processed/Sports_and_Outdoors_mappings.json` | no | `/data/mappings.json` |
| `AGENTRECX_DATA_DIR` | data root for derived paths | `<repo>/data` | no | `/mnt/agentrecx-data` |
| `AGENTRECX_MEMORY_DB` | preference-memory SQLite file | `data/artifacts/demo/preference_memory.sqlite3` | no | `/tmp/demo/memory.sqlite3` |
| `AGENTRECX_DEVICE` | inference device | `cpu` | no | `cuda:0` |
| `AGENTRECX_HOST` | server bind host | `127.0.0.1` | no | `0.0.0.0` |
| `AGENTRECX_PORT` | server bind port | `8000` | no | `8011` |
| `AGENTRECX_VERIFY_CHECKPOINT` | verify checkpoint digest at startup | `1` (enabled) | no | `0` to disable |

Notes:

* `AGENTRECX_VERIFY_CHECKPOINT` is treated as false only for `0`, `false` or `no`
  (case-insensitive).
* `AGENTRECX_DATA_DIR` is read when the configuration module is imported, so set it before
  starting a process rather than at runtime.
* No variable accepts a secret; the demo reads no credentials.

---

## Running the Web Demo

```bash
cd /root/AgentRec-X
.venv/bin/python -m recommendation.api.app --host 127.0.0.1 --port 8000
```

Startup loads the accepted checkpoint and the catalogue metadata once. Expect roughly ~20 s
on first start (checkpoint + 300 MB metadata parse) and an `Application startup complete.`
line. If an artifact is missing the process exits with an explicit error instead of starting
a broken server.

CLI flags (verified in the argparse definition):

| Flag | Meaning | Default |
| --- | --- | --- |
| `--host` | bind host | `AGENTRECX_HOST` or `127.0.0.1` |
| `--port` | bind port | `AGENTRECX_PORT` or `8000` |
| `--device` | `cpu` / `cuda` / `cuda:0` | `AGENTRECX_DEVICE` or `cpu` |
| `--checkpoint` | override checkpoint path | `AGENTRECX_CHECKPOINT_PATH` |
| `--mappings` | override mappings path | `AGENTRECX_MAPPINGS_PATH` |
| `--manifest` | override manifest path | `AGENTRECX_MANIFEST_PATH` |
| `--reload` | uvicorn auto-reload (development) | off |

To bind a different port, or to keep the demo's memory database out of the repository:

```bash
AGENTRECX_MEMORY_DB=/tmp/demo/memory.sqlite3 \
  .venv/bin/python -m recommendation.api.app --host 127.0.0.1 --port 8011
```

Stop the server with `Ctrl+C`. This is a local research demo: bind to `127.0.0.1` for
personal use, and if a remote host needs access, forward the port (for example an SSH
tunnel) rather than exposing the unauthenticated API to a network.

---

## Opening the browser UI

| Page | URL |
| --- | --- |
| Demo | `http://127.0.0.1:8000/demo/` |
| Root redirect | `http://127.0.0.1:8000/` → `/demo/` |
| Interactive API docs | `http://127.0.0.1:8000/docs` |
| Demo readiness | `http://127.0.0.1:8000/v1/demo/health` |

The page loads only same-origin assets — no CDN, no npm, no build step.

---

## Using the demo

1. **Session starts automatically.** On load the page probes readiness, fetches the demo
   profiles, and creates a session for the first profile (`demo-user-1`). The
   *System / audit* panel shows the state (`initializing` → `ready`), the session id, the
   turn number and the route.
2. **Ask for recommendations.** Type e.g. `Recommend some products.` and press **Send**.
   Cards appear in the exact order the backend produced.
3. **Read a card.** Each shows the final rank, the product title (`Title unavailable` when
   the catalogue has no record), store / category / price when available, the
   **original SASRec rank**, the **raw SASRec ranking score** (explicitly not a probability),
   and the preference-evidence counts.
4. **State a preference.** Type e.g. `I don't want red.` The banner reports
   *"Preference saved for future turns: avoid red"*, and the *Active preferences* panel
   updates. The audit panel still shows `0 preference(s)` under **Ranked with**, because
   this turn was ranked with the snapshot taken before the write.
5. **Ask again.** Type `Recommend again.` Now **Ranked with** shows `1 preference(s)`, and
   cards may be reordered. A moved card carries a factual line such as
   *"Moved from rank 4 to rank 1 under the explicit-preference policy."*
6. **Replace or remove a preference.** `Actually, I prefer blue instead.` supersedes the
   previous value; `I don't care about color anymore.` removes the colour constraint, and
   the panel empties.
7. **Inspect the order.** The collapsible *Candidate order (audit)* section lists the
   original SASRec order and the order after the policy.
8. **Reset.** **Reset session** deletes the current session (its id stops working) and starts
   a fresh one. **New session** creates an additional isolated session.

Reading the ranks correctly:

| Field | Meaning |
| --- | --- |
| **Original SASRec rank** | position the recommender gave the candidate |
| **Final rank / reranked rank** | position after the explicit-preference policy |
| **Raw SASRec ranking score** | ordering value from the model; **not** a probability, confidence, rating or preference score |
| **match / violation / unknown** | M10A evidence counts. `unknown` means the metadata cannot decide — it is **not** a failed match |
| `Moved from rank X to rank Y` | the two ranks actually produced. No quality claim is attached |

The preference panel shows **ACTIVE** preferences only; superseded and removed entries are
retained for audit but are not displayed as current.

---

## Example multi-turn conversation

The following is a **verified** transcript against the accepted artifacts with the
`demo-user-1` profile (the first eligible user in accepted sequence order). Session ids and
timings are omitted.

```text
User:  Recommend some products.

Agent: Top 5 candidate(s) from the sequential recommender, ordered by the configured
       explicit-preference policy (...), with catalogue facts where available:
       1. B0BX5QFWQN (original SASRec rank 1, ranking score +5.8085)
       ...
       ranked with 0 preference(s)
```

```text
User:  I don't want red.

System: Preference saved for future turns: avoid red
        active preferences: [color avoid red]
        ranked with 0 preference(s)      <- this turn used the pre-turn snapshot
```

```text
User:  Recommend again.

Agent: ... ranked with 1 preference(s)
```

On this particular candidate set, `avoid red` alone does **not** reorder anything, because
none of the five candidates has a decisive colour violation — their colour evidence is
`UNKNOWN`. That is the accepted conservative asymmetry, not a failure: a readable field
holding a different value is deliberately `UNKNOWN` rather than a violation.

Adding more explicit constraints does produce movement. With the fixed synthetic preference
fixture used by the M11 smoke (`I don't want red.` / `I don't want blue.` /
`I prefer black.` / `I prefer lightweight hiking gear.` / `My budget is under $100.`), the
same profile and the same five candidates reorder as follows:

```text
original : B0BX5QFWQN, B0BBFB48YQ, B00C6OUDX2, B0855B4QZR, B01L6RE7Z4
reranked : B0855B4QZR, B0BX5QFWQN, B00C6OUDX2, B0BBFB48YQ, B01L6RE7Z4
moved    : 3

  orig -> rerank   parent_asin   sasrec_score   match  viol  unk
     4 -> 1        B0855B4QZR      +5.5415        2     0     3
     1 -> 2        B0BX5QFWQN      +5.8085        1     0     4
     3 -> 3        B00C6OUDX2      +5.6675        1     0     4
     2 -> 4        B0BBFB48YQ      +5.6839        0     0     5
     5 -> 5        B01L6RE7Z4      +5.4344        0     0     5
```

This exact output is reproduced by the M11 smoke, and it matches the M10D agent-level
result for the same profile and fixture. The preference fixture is synthetic and is used for
engineering validation only.

---

## Calling the HTTP API directly

All examples assume the server is on `127.0.0.1:8000`.

### Preserved Milestone 6 endpoints

```bash
# readiness
curl -s http://127.0.0.1:8000/health

# model metadata
curl -s http://127.0.0.1:8000/v1/model

# direct top-k inference (NOT routed through the agent)
curl -s -X POST http://127.0.0.1:8000/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"history": ["B0BX5QFWQN", "B0BBFB48YQ", "B00C6OUDX2"], "k": 3}'
```

### Demo endpoints

```bash
# demo readiness
curl -s http://127.0.0.1:8000/v1/demo/health

# server-owned demo profiles
curl -s http://127.0.0.1:8000/v1/demo/profiles

# create a session -> returns session_id
curl -s -X POST http://127.0.0.1:8000/v1/demo/sessions \
  -H 'Content-Type: application/json' \
  -d '{"profile_id": "demo-user-1"}'

# session state (metadata, turn count, ACTIVE preferences)
curl -s http://127.0.0.1:8000/v1/demo/sessions/<SESSION_ID>

# one turn: ask for recommendations
curl -s -X POST http://127.0.0.1:8000/v1/demo/sessions/<SESSION_ID>/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Recommend some products.", "k": 5}'

# one turn: state a preference (saved for FUTURE turns)
curl -s -X POST http://127.0.0.1:8000/v1/demo/sessions/<SESSION_ID>/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "I don'"'"'t want red.", "k": 5}'

# reset the session (only that session; its id stops working)
curl -s -X DELETE http://127.0.0.1:8000/v1/demo/sessions/<SESSION_ID>
```

A chat response is structured, not prose-only. Top-level keys:

```text
api_version  session_id  turn_id  turn  route  message
active_preferences   memory_update   recommendations[]   audit
```

Each `recommendations[]` entry carries `reranked_rank`, `original_rank`, `parent_asin`,
`item_id`, `sasrec_score`, `metadata_status`, `metadata`, `match_count`, `violation_count`,
`unknown_count`, `evidence[]`, `fallback_reason` and `movement_summary`.

Validation and error behaviour:

* `message` must be a non-blank string of at most 2000 characters;
* `k` must be a **strict** integer in `1..100` (so `"5"` and `5.0` are rejected);
* request bodies forbid extra fields — sending `trusted_user_history`, `history`,
  `parent_asins`, `preference_snapshot`, `reranking`, `user_key`, `session_id`, `turn_id`
  or `route` returns **422**;
* unknown/expired/reset session → **404** `session_not_found`;
* unknown profile → **404** `unknown_profile`;
* session capacity reached → **503** `session_capacity_exceeded`;
* backend failures (recommendation, preference stage, agent) → **502** with an
  authored detail. A failed turn never returns fabricated recommendation content.

---

## Running SASRec recommend endpoint

```bash
curl -s http://127.0.0.1:8000/v1/model
curl -s -X POST http://127.0.0.1:8000/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"history": ["B0BX5QFWQN", "B0BBFB48YQ", "B00C6OUDX2"], "k": 5}'
```

The `score` field is a raw SASRec ranking score. History items must exist in the served
catalogue: an unknown `parent_asin` produces `422 unknown_item` rather than being dropped,
and an empty history produces `422` — there is no cold-start fallback.

---

## Running tests

```bash
cd /root/AgentRec-X
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q recommendation tests experiments
```

Expected: **1372 passed, 2 skipped** (~46 s) in the working tree that includes the
documentation-verification tests; the last milestone commit (`c7c3bf5`) contained
**1315 passed, 2 skipped** before those 57 tests were added. Suites needing the accepted
checkpoint or the metadata artifact skip cleanly when those files are absent.

Useful subsets:

```bash
.venv/bin/python -m pytest -q tests/test_api.py                 # M6 API contract (27 tests)
.venv/bin/python -m pytest -q tests/test_agent_tool_e2e.py      # M7C real chain (36 tests)
.venv/bin/python -m pytest -q tests/test_demo_api.py tests/test_demo_multiturn.py
.venv/bin/python -m pytest -q tests/test_memory_store.py tests/test_memory_service.py
```

If your environment exports `OMP_NUM_THREADS` as an empty or non-positive value, libgomp
rejects it on torch import. The repository's `conftest.py` normalises it for pytest; for
direct script runs, export a positive value first (see Troubleshooting).

---

## Running smoke tests

Smoke scripts live in `experiments/`, print explicit PASS/FAIL gates, and exit non-zero on
failure. All have `--json <path>` for a machine-readable envelope; `--k` selects the
candidate count where relevant.

| Smoke | Milestone | Notes |
| --- | --- | --- |
| `experiments.recommendation_tool_smoke` | M7A | Tool contract; flags `--calls --histories --checkpoint --mappings --manifest --device --k --json` |
| `experiments.agent_graph_smoke` | M7B | offline orchestration; `--decision-json --k --json` |
| `experiments.agent_tool_e2e_smoke` | M7C | real checkpoint; `--checkpoint --device --k --json` |
| `experiments.product_rag_smoke` | M8 | metadata/RAG; `--artifact --query --repeats --device --k --json` |
| `experiments.agent_product_rag_smoke` | M8 | agent + real RAG; `--query --device --k --json` |
| `experiments.memory_smoke` | M9 | preference memory; `--device --k --json --skip-real` |
| `experiments.preference_matching_smoke` | M10A | evidence; `--device --k --json --skip-real` |
| `experiments.preference_reranking_smoke` | M10B | reranking policy; `--device --k --json --skip-real` |
| `experiments.reranking_evaluation_smoke` | M10C | policy diagnostics; `--cohort --device --k --json --skip-real` |
| `experiments.agent_reranking_smoke` | M10D | agent + reranking; `--query --device --k --json` |
| `experiments.web_demo_smoke` | M11 | full demo over HTTP; `--device --k --json` |

Run the three most informative ones:

```bash
.venv/bin/python -m experiments.web_demo_smoke             # M11, real HTTP, 39 gates
.venv/bin/python -m experiments.agent_reranking_smoke      # M10D, real chain
.venv/bin/python -m experiments.reranking_evaluation_smoke # M10C, policy diagnostics
```

Other real-chain smokes:

```bash
.venv/bin/python -m experiments.sasrec_inference_smoke
.venv/bin/python -m experiments.agent_tool_e2e_smoke
.venv/bin/python -m experiments.agent_product_rag_smoke
.venv/bin/python -m experiments.memory_smoke
.venv/bin/python -m experiments.preference_matching_smoke
.venv/bin/python -m experiments.preference_reranking_smoke
```

Add `--skip-real` to the M9/M10A/M10B/M10C smokes to run only their synthetic portions
without the checkpoint or the 300 MB metadata artifact.

---

## Running selected milestone checks

Every smoke above is a milestone check. A representative full sweep (verified to exit 0 on
the accepted artifacts):

```bash
for m in experiments.recommendation_tool_smoke experiments.agent_graph_smoke \
         experiments.agent_tool_e2e_smoke experiments.agent_product_rag_smoke \
         experiments.memory_smoke experiments.preference_matching_smoke \
         experiments.preference_reranking_smoke experiments.reranking_evaluation_smoke \
         experiments.agent_reranking_smoke experiments.web_demo_smoke; do
  printf '%-48s' "$m"
  .venv/bin/python -m "$m" > "/tmp/$(basename "$m").log" 2>&1 && echo PASS || echo FAIL
done
```

Inspect a log for its gate list:

```bash
grep -E '^(SMOKE|  PASS|  FAIL)' /tmp/web_demo_smoke.log
```

Cost classes:

| Task | Cost |
| --- | --- |
| `pip check`, `compileall`, offline test subsets | seconds |
| Full pytest suite | ~45 s |
| M11 smoke (real checkpoint + metadata + HTTP) | ~1 min |
| Rebuilding the metadata artifact | ~1 min plus parsing the 1 GB raw file |
| Re-training SASRec on full data | hours on GPU — see [EXPERIMENTS.md](EXPERIMENTS.md) |

Full-dataset preprocessing and SASRec training are documented in
[EXPERIMENTS.md](EXPERIMENTS.md#15-reproducibility). They are **not** part of Quick Start
and they write to new run directories rather than overwriting the accepted artifacts:

```bash
# labelled for completeness; NOT executed as part of this documentation task
.venv/bin/python -m experiments.sasrec_canonical --canonical   # new run directory, GPU
.venv/bin/python -m experiments.sasrec_formal_test            # opens the sealed test set
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: accepted sequences artifact not found` / server exits at startup | An accepted artifact is missing or the path is wrong | Check the paths in [Required artifacts](#required-artifacts); override with `AGENTRECX_*` variables |
| `libgomp: Invalid value for environment variable OMP_NUM_THREADS` on any torch import | The environment exports `OMP_NUM_THREADS` as empty or `0` | `export OMP_NUM_THREADS=8` before running a script directly (pytest is handled by `conftest.py`) |
| `404 session_not_found` after a while, or after a server restart | The live session registry is process-local and does not survive a restart; sessions can also expire when a TTL is configured | Create a new session; stored preference memory is unaffected |
| `503 demo_unavailable` | The demo runtime was not composed | Read the startup log; check `GET /v1/demo/health` for `model_loaded` / `metadata_loaded` / `demo_ready` |
| `503 session_capacity_exceeded` | The bounded live-session registry is full (default 64) | Reset sessions you no longer need, then create a new one |
| `422 invalid_request` | Blank or over-long `message`, non-strict or out-of-range `k`, or an extra field in the body | Send only `{"message": "...", "k": 5}` |
| Port already in use | An earlier server is still running | Use `--port <free>` or stop the other process |
| `pip check` reports a conflict | Something changed the accepted HTTP stack (`httpx`/`h11`) | Reinstall from `requirements.txt`; the `langsmith` pin in that file exists to keep this closure satisfiable |
| Server starts but `/v1/model` returns 500 | The model failed to load; `/health` reports `model_loaded: false` with a reason | Fix the reported artifact problem rather than the endpoint |
| Recommendation smoke prints `SKIP` | The checkpoint/metadata artifact is absent | Expected on a fresh clone; see [Required artifacts](#required-artifacts) |

For a quick health triage:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/demo/health
```
