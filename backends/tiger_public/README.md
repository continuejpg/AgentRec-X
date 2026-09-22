# Public-TIGER backend (Step 2.3 skeleton)

A generative-retrieval backend that AgentRec-X consumes through an explicit adapter boundary.
It owns **learning and scoring only**; it owns none of the evaluation.

Specification: [`docs/TIGER_BACKEND.md`](../../docs/TIGER_BACKEND.md).
Boundary rule: [`AGENTS.md`](../../AGENTS.md) section 19.

> **Status: skeleton only.** Every stage in `src/tiger_public/cli.py` writes a deterministic
> *placeholder* labelled `step-2.3-placeholder-no-ml` in the artifact it produces. There is no
> item encoder, no RQ-VAE, no Semantic-ID training, no TIGER model, no constrained retrieval and
> no measurement. Nothing here trains, and nothing here touches a GPU.

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
build-features  -> ItemFeatureArtifact     (Step 2.4 replaces the placeholder)
fit-sid         -> SemanticIdArtifact      (Step 2.4 replaces the placeholder)
train           -> TigerCheckpoint         (Step 2.5 replaces the placeholder)
score           -> Iterator[ScoreBatch]    (Step 2.6 replaces the stub scorer)
```

Production crossing is a subprocess plus a filesystem contract (JSONL / `.npy` / `.npz` /
`.json`), driven from AgentRec-X by `recommendation/backends/tiger_backend.py`.

## Running it

The Step-2.3 placeholders need only NumPy, so they run in either environment; a real deployment
uses the backend's own venv.

```bash
# backend-local tests (its own project; never collected by AgentRec-X)
cd backends/tiger_public
PYTHONPATH=src python -m pytest tests -q

# drive the whole boundary end to end from AgentRec-X (writes to /tmp)
.venv/bin/python -m experiments.materialize_tiger_backend --limit 2000 \
    --out /tmp/tiger_handoff --cohort 64 --mode stub-run --status approximate
```

`--limit N` shrinks the catalogue for a smoke run only; the cohort definition, the split, the
cohort seed and the evaluation protocol are never altered by it.

## Layout

```text
src/tiger_public/
  contracts.py   frozen dataclasses, the PAD sentinel, the token layout, the Protocol
  io.py          hashing, manifest verification, handoff readers, forbidden-key refusal
  scoring.py     the ONE frozen item-score rule and its admissibility helper
  cli.py         the four stage subcommands (all placeholders in Step 2.3)
tests/           backend-local invariants
PROVENANCE.md    per-file origin and licence status
```
