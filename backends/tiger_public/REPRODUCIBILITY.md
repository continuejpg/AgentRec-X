# Public-TIGER backend — reproducible ML environment

The backend runs in its **own** virtual environment. Never install these into AgentRec-X's
environment and never add them to the root `requirements.txt`: `AGENTS.md` sections 10-11 forbid
disturbing the accepted CPU PyTorch stack, and the backend needs a CUDA build for the
production run.

```bash
cd backends/tiger_public
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements-ml.txt
```

## Two separate decisions

Package versions and the CUDA wheel are deliberately kept apart. A pinned version list is a
statement about *APIs*; a wheel index is a statement about *the host*. Choosing the second one
on this development box would be guessing about the AutoDL machine, so it is not done here.

### 1. Package versions

`requirements-ml.txt` pins the versions **actually exercised by the Step-2.4 runs** on this
host, with `pyarrow` carried for the catalogue tooling. `torch` is intentionally absent from
that file — see below.

| Package | Version demonstrated | Notes |
|---|---|---|
| Python | 3.10.12 | `pyproject.toml` requires `>=3.10` |
| numpy | 1.26.4 | `numpy<2` is also what `pyproject.toml` declares |
| torch | 2.14.0+cpu | CPU build used locally; the production host needs a CUDA build |
| transformers | 5.17.0 | loads `sentence-t5-base` without a `trust_remote_code` path |
| sentence-transformers | 6.1.0 | `get_embedding_dimension()`; `get_sentence_embedding_dimension()` is a deprecated alias |
| pyarrow | (catalogue tooling) | declared, not exercised by Step 2.4 |

Two API notes worth keeping, because both were observed during Step 2.4 rather than assumed:

* `sentence-transformers` 6.x returns **float16** from `encode` by default; the feature stage
  immediately casts to `float32`, so the artifact dtype does not depend on this.
* `_commit_hash` is **no longer set** on a loaded model in 6.x. The revision must therefore come
  from `--encoder-revision` or from the Hub cache's `refs/main`, which is why
  `features.is_immutable_revision` treats a non-SHA value as "unpinned" rather than trusting it.

### 2. CUDA wheel / index selection

Not yet chosen. The production torch build must be picked **on the AutoDL host** once its driver
and CUDA runtime are known:

```bash
nvidia-smi                       # driver + CUDA runtime the host actually has
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Substitute the `cuXXX` matching that host. A wheel chosen here and shipped blind could be
incompatible with the host's driver, and the failure mode — a silent CPU fallback — would only
show up as an inexplicable 15-hour encode.

## Revision pinning

The encoder's weights are pinned by **immutable snapshot SHA-1**, not by repository name:

```bash
--encoder-revision fc5d4628481afbbaaacd7af6bb07cf9d3865f781
```

That SHA is the snapshot the successful Step-2.4 run used, recovered from the local Hub cache
(`refs/main` and the snapshot directory name both carry it). Passing it makes the loader fetch or
resolve that exact commit, and the feature artifact records three distinct things:

```text
encoder.requested_revision   what was asked for (may be null)
encoder.revision             the resolved commit, or null when unresolved
encoder.weights_pinned       true only when a requested revision resolved to a commit
```

With `--encoder-revision` set, a run that cannot confirm the pin **fails** rather than recording
a guarantee it does not have. Adding `--local-files-only` turns a missing snapshot into an
immediate offline error instead of a network fetch.

## Provenance recorded by every production run

`build-features` and `fit-sid` write an `environment` block into their `manifest.json` via
`environment_metadata()`, containing exactly:

```text
python, numpy, torch, torch CUDA version, CUDA availability, GPU name,
transformers, sentence_transformers, backend git commit, device,
deterministic_algorithms
```

Where a value is unavailable it is recorded as unavailable (`"not installed"`,
`"unavailable"`), never omitted and never estimated.
