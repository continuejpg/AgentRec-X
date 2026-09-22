# Public-TIGER backend (Step 2.3 skeleton)

A generative-retrieval backend that AgentRec-X consumes through an explicit adapter boundary.
It owns **learning and scoring only**; it owns none of the evaluation.

Specification: [`docs/TIGER_BACKEND.md`](../../docs/TIGER_BACKEND.md).
Boundary rule: [`AGENTS.md`](../../AGENTS.md) section 19.

> **Status after Step 2.4.** Stages 1 and 2 are implemented:
>
> * `build-features` encodes `products_text.jsonl` with a frozen sentence encoder;
> * `fit-sid` trains the RQ-VAE, assigns three-level Semantic IDs, applies the deterministic
>   dedup digit, and audits the result.
>
> Stage 3 (`train`) and stage 4 (`score`) remain **placeholders** labelled
> `step-2.3-placeholder-no-ml`: there is no TIGER model, no constrained retrieval, no certified
> search and no recommendation measurement in this tree yet.

## Why this is a separate project

| Reason | Consequence |
| --- | --- |
| The backend needs a CUDA PyTorch wheel; AgentRec-X runs a CPU PyTorch build | its own virtual environment; the root `requirements.txt` is never modified |
| An external repository's licence may forbid vendoring | absorbed elements are reimplemented and labelled in `PROVENANCE.md` |
| AgentRec-X must stay runnable without any backend dependency | `pytest.ini` sets `norecursedirs = backends`, so the AgentRec-X suite never collects these tests |

## What the boundary guarantees

The backend never receives, and cannot reconstruct:

* **canonical identity** — it is given opaque integer `item_id` values and nothing else. No
  `parent_asin` appears as an identifier, a field or an artifact entry anywhere in this tree.
* **evaluation targets** — the cohort handoff carries `case_id`, `history` and an integer
  `required_frontier`. There is no `test_target`, no `validation_target`, and no "grade-only"
  switch that would imply a protected field exists.
* **the seen set** — it receives `required_frontier = K_max + |seen|`, a count. It cannot mask
  and it cannot reconstruct which items were seen.
* **ranking and metrics** — it returns raw scores. Seen-item masking, tie-breaking, ranking and
  Recall/NDCG/HR all belong to AgentRec-X's frozen evaluator.

## The four stages

```text
build-features  -> ItemFeatureArtifact     IMPLEMENTED  frozen text encoder -> [N, 768] float32
fit-sid         -> SemanticIdArtifact      IMPLEMENTED  RQ-VAE -> 3-level SIDs -> dedup digit
train           -> TigerCheckpoint         placeholder  (Step 2.5)
score           -> Iterator[ScoreBatch]    placeholder  (Step 2.6)
```

### Stage 1 — `build-features`

Reads **only** `products_text.jsonl`, so it cannot reach the original catalogue or canonical
identity.  Rows are encoded in batches into a memory-mapped array and the file is **renamed
into place only on success**: an interrupted build leaves no artifact, because an in-place write
would leave a complete-looking file whose unwritten rows are zeros and a zero vector is a valid
embedding that nothing downstream would question.  Items with empty text are still embedded, so
coverage stays `1.000`, and any non-finite vector is replaced by the batch mean and counted.

`--encoder smoke` selects a deterministic character-ngram stand-in for plumbing-only runs on a
machine with no GPU and no network.  It is labelled
`step-2.4-smoke-encoder-not-sentence-t5` in the artifact, so nothing measured with it can be
mistaken for a Sentence-T5 result.

### Stage 2 — `fit-sid`

```text
features [N, 768] -> normalise -> encoder 768-256-128-64 -> 3 residual codebooks of 256
                  -> codes -> audit -> dedup ordinal -> SemanticIdArtifact
```

The order of the guards is part of the design: the handoff and feature artifacts are verified
first (including that the features were built from *this* `products_text.jsonl`), then the
quantizer trains, then the dead-code stop, then the **pre-dedup** collision statistics are
computed, then the dedup pass runs, and only then is anything written.

`--allow-dead-codes` waives the >20 % dead-code stop for a small-scale mechanism check.  The
waiver is recorded in `semantic_ids.json` under `dead_code_waiver`, so it cannot be silently
inherited by a full run.

`fit-sid` is **transactional**: every artifact is written into a sibling `*.partial` staging
directory, the manifest is written last and re-verified from disk, and only then is the
directory promoted with a single rename.  An interrupted run therefore leaves either an
explicitly incomplete `*.partial` directory or nothing — never a final directory that could be
mistaken for a completed run.

A production run should also pass `--encoder-revision <sha>` to `build-features`, and must not
pass `--no-require-features-manifest` to `fit-sid` (which exists only so older fixtures keep
working and is recorded in the artifact when used).

### Learning rate

The Step-2.4 measurement found that the published `learning_rate = 1e-3` collapses levels 1 and
2 of the residual hierarchy (236 and 240 of 256 codes dead on the smoke catalogue).  The CLI
therefore defaults to `1e-3` — the frozen specification value — and the smoke runner passes
`--learning-rate 3e-4`, which is the measured setting reported in the Step-2.4 completion
report.  See `docs/TIGER_BACKEND.md` section 5 and the report for the numbers; the choice is
frozen there, not tuned against any evaluation cohort.

Production crossing is a subprocess plus a filesystem contract (JSONL / `.npy` / `.npz` /
`.json`), driven from AgentRec-X by `recommendation/backends/tiger_backend.py`.

## Running it

The Step-2.3 placeholders need only NumPy, so they run in either environment; a real deployment
uses the backend's own venv.

```bash
# backend-local tests (its own project; never collected by AgentRec-X)
cd backends/tiger_public
PYTHONPATH=src .venv/bin/python -m pytest tests -q

# the Step-2.4 smoke gate: features + RQ-VAE + dedup, on a small item subset
cd ../..
.venv/bin/python -m experiments.smoke_tiger_step24 --items 512 --encoder smoke

# a real run needs the encoder's weights, which a GPU host fetches from the Hub or a mirror
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m experiments.smoke_tiger_step24 \
    --items 2000 --encoder sentence-transformers/sentence-t5-base
```

`--limit N` remaps the cohort onto its N most-used items and carries the real AgentRec-X item
ids through under a new `backend_row` order; the cohort definition, the split, the cohort seed
and the evaluation protocol are never altered by it.

The **full-catalogue** stage-1 run is a GPU job: this backend encodes at roughly 3 items/s on a
CPU-only host, i.e. about 15 hours for 156 746 items, against minutes on an RTX 4090.  Do not
compromise the model to fit a CPU.

## Layout

```text
src/tiger_public/
  contracts.py   frozen dataclasses, the PAD sentinel, the token layout, the Protocol
  io.py          hashing, manifest verification, handoff readers, forbidden-key refusal
  scoring.py     the ONE frozen item-score rule and its admissibility helper
  features.py    stage 1 - text encoder, streaming batches, atomic artifact write
  quantizer.py   stage 2 - RQ-VAE, k-means++ seeding, STE, dead-code revival, diagnostics
  dedup.py       stage 2 - deterministic collision ordinal with an overflow refusal
  cli.py         the four stage subcommands
tests/           backend-local invariants
PROVENANCE.md      per-file origin and licence status
REPRODUCIBILITY.md the pinned ML runtime and the encoder snapshot pin
requirements-ml.txt the pinned package versions (torch installed separately, per host)
```

### Read-only diagnostics

`experiments/audit_tiger_sid.py` reports what the aggregate `reconstruction_loss` cannot: the
prefix reconstruction MSE at L0 / L0+L1 / L0+L1+L2, residual norms after each level, per-level
occupancy histograms, coverage, entropy, the **largest-code fraction** (a different question from
coverage), and the pre-dedup collision-group size distribution.  It trains nothing and modifies
nothing.
