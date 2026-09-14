# Recommendation API (Milestone 6)

A minimal HTTP service that turns the accepted Milestone 5 SASRec checkpoint into
recommendations. It is a thin adapter: all model, mapping, encoding, masking and
ranking logic lives in [`../inference/`](../inference/) (`recommendation/inference/`)
and is shared with future non-HTTP callers.

```
HTTP (FastAPI, this package)
        ↓
SASRecInferenceEngine        recommendation/inference/sasrec.py
        ↓
deterministic ranking        recommendation/inference/ranking.py
        ↓
SASRec (eval, inference_mode)
```

There is **no** Agent, RAG, Memory, Critic, conversation state or LLM call here, and no
authentication, database or user accounts.

---

## 1. Artifact requirements

| Artifact | Purpose |
| --- | --- |
| `runs/sasrec_canonical_2026/best.pt` | accepted SASRec checkpoint (SHA-256 `352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912`) |
| `runs/sasrec_canonical_2026/run.json` | formal run manifest, cross-checked against the checkpoint |
| `data/processed/Sports_and_Outdoors_mappings.json` | `parent_asin <-> int id` mapping and catalog size |

At startup the loader verifies: checkpoint format, checkpoint SHA-256 (when
`expected_checkpoint_sha256` is configured), manifest/checkpoint agreement on
`num_items`, `max_seq_len`, `hidden_size`, `num_blocks`, `num_heads`, `dropout`, and
mapping cardinality/contiguity (`1..num_items`, PAD unused). Any mismatch **fails
startup** — embeddings are never resized and incompatible keys are never ignored.

## 2. Install and run

```bash
.venv/bin/python -m pip install -r requirements.txt      # fastapi, uvicorn, pydantic

# development server (local binding by default)
.venv/bin/python -m recommendation.api.app --host 127.0.0.1 --port 8000
# or
.venv/bin/uvicorn recommendation.api.app:app --host 127.0.0.1 --port 8000
```

The model is loaded **once** during startup (never per request). Startup also warms a
single inference path. If loading fails the process still runs but reports
`model_loaded: false`, and `/v1/recommend` and `/v1/model` return 500.

### Configuration

CLI flags override environment variables, which override repository-layout defaults.

| Setting | Environment variable | Default |
| --- | --- | --- |
| checkpoint path | `AGENTRECX_CHECKPOINT_PATH` | `runs/sasrec_canonical_2026/best.pt` |
| manifest path | `AGENTRECX_MANIFEST_PATH` | `runs/sasrec_canonical_2026/run.json` |
| mapping path | `AGENTRECX_MAPPINGS_PATH` | `data/processed/Sports_and_Outdoors_mappings.json` |
| device | `AGENTRECX_DEVICE` | `cpu` |
| host / port | `AGENTRECX_HOST` / `AGENTRECX_PORT` | `127.0.0.1` / `8000` |
| checkpoint identity check | `AGENTRECX_VERIFY_CHECKPOINT` | `1` |

### Device

`cpu` (default), `cuda` and `cuda:0` are accepted. Requesting CUDA when
`torch.cuda.is_available()` is false **fails startup** rather than silently serving on
CPU. The active device is reported by `/health` and `/v1/model`.

## 3. Endpoints

### `GET /health`

Distinguishes "process running" from "model loaded".

```json
{"status": "ok", "model_loaded": true, "device": "cpu", "detail": null}
```

When the model failed to load: `status: "unavailable"`, `model_loaded: false`,
`device: null`, and `detail` explains the failure.

### `GET /v1/model`

```json
{
  "model_type": "SASRec",
  "num_items": 156746,
  "max_seq_len": 50,
  "hidden_size": 64,
  "num_blocks": 2,
  "num_heads": 2,
  "dropout": 0.2,
  "device": "cpu",
  "checkpoint_sha256": "352bd3ae…",
  "parameter_count": 10135104,
  "model_parameters_frozen": true,
  "provenance": {
    "formal_run_git": {"commit": "859fd0bb…", "branch": "master", "dirty": true, "note": "…"},
    "serving_note": "…"
  }
}
```

**Provenance is deliberately split.** `formal_run_git` is the Git state the formal
training run recorded (`859fd0bb…`, dirty). The post-run accepted source checkpoint is
the later commit `5244298137f8b70e516b4846d905ab559834cdfe`. The model was **not**
trained from that later commit; the checkpoint simply captures the reviewed source tree
corresponding to the accepted implementation. No local filesystem path is exposed.

### `POST /v1/recommend`

Request:

```json
{"history": ["B00EXAMPLE1", "B00EXAMPLE2"], "k": 10}
```

| Field | Rules |
| --- | --- |
| `history` | non-empty list of `parent_asin` strings, chronological. Duplicates are allowed (repeated interactions are real). |
| `k` | integer, `1..100`, default `10` |

Response:

```json
{
  "recommendations": [
    {"rank": 1, "item_id": 151270, "parent_asin": "B0BX5QFWQN", "score": 5.8085}
  ],
  "requested_k": 5,
  "returned_k": 5,
  "history_length": 5,
  "effective_history_length": 5,
  "history_truncated": false,
  "eligible_candidates": 156741,
  "timings_ms": {"scoring": 14.8, "ranking": 18.8}
}
```

`returned_k = min(requested_k, eligible_candidates)`. If nothing is eligible, the list
is empty with `returned_k: 0` — never an error, never fabricated candidates.

## 4. Contract details

### Score meaning

> The returned `score` is a raw **SASRec model score**, not a calibrated probability.

It is not a probability, confidence, or purchase likelihood; it is only meaningful for
ordering candidates. No product metadata (title, price, brand, description, image,
category) is invented or returned — only `parent_asin` and the model score.

### Item identity

External identity is the Amazon `parent_asin`. Internally the model uses integer ids
`1..num_items`, with `0` reserved for PAD. The mapping comes from the existing
preprocessing artifact; no second mapping system exists.

### Unknown items

Unknown `parent_asin` values are **rejected** with HTTP 422 (`error: "unknown_item"`).
They are never coerced to PAD and never silently dropped.

### History truncation

The model window is the newest `max_seq_len = 50` items, left-padded with PAD when the
history is shorter, preserving chronological order. Longer histories keep the last 50;
`history_truncated` and `effective_history_length` report this.

### Seen-item filtering

Every item in the **entire supplied history** is excluded from the results — not just
the 50 that reached the model. The model sees a truncated window; the ranking layer
knows the full history.

### Ranking

Candidates are `1..num_items` minus the seen items (PAD is never eligible). Order:
higher score first; on an exact score tie, the **lower integer item id** wins. This is
enforced with an explicit lexicographic sort key, not by relying on `topk` stability.

### Errors

| Condition | Status | `error` |
| --- | --- | --- |
| empty history, `k` out of range, wrong types, malformed JSON, unknown body field | 422 | `invalid_request` |
| unknown `parent_asin` | 422 | `unknown_item` |
| model not loaded / internal failure | 500 | `inference_failed` |

Error bodies never contain stack traces, local paths or environment secrets.

## 5. Inference and concurrency

Every request runs `model.eval()` under `torch.inference_mode()`. No gradients are
created, no optimizer exists, and weights never change. Requests are served
sequentially by a single process; Python-level request handling is not parallelised
inside the engine. A single uvicorn worker is the intended deployment for this
milestone — scaling out would mean multiple processes, each loading its own copy of
the model.

Per request the engine encodes one history, runs one forward pass, masks, ranks, and
discards the full score vector. Per-user score matrices are never cached.

## 6. Smoke and latency

```bash
.venv/bin/python -m experiments.sasrec_inference_smoke
.venv/bin/python -m experiments.sasrec_inference_smoke --json /tmp/inference.json
```

Prints device, checkpoint identity, catalog size, history length/truncation, top-k
`parent_asin`s and CPU latency percentiles. It loads the real accepted checkpoint; the
synthetic fixtures used by the test suite are described below.

## 7. Tests

```bash
.venv/bin/python -m pytest tests/test_sasrec_ranking.py tests/test_sasrec_inference.py tests/test_api.py -q
```

The ranking and inference suites use an explicit full-sort oracle, so the vectorised
ranking cannot drift from the documented rule. The API suite drives FastAPI's
in-process `TestClient` against a tiny synthetic checkpoint
(`tests/sasrec_inference_fixture.py`); normal pytest never loads the 349 MB formal run
or the 156,746-item catalog. All suites run on CPU.
