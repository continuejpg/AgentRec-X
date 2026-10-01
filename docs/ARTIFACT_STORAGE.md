# Large-artifact storage and verification

M1 deliberately does NOT commit large artifacts. The repository holds code, tests, experiment
scripts, configuration, small manifests and accepted reports only. Every large artifact below is
recorded by SHA256, by the location actually verified during the M1 durability audit, and by its
producer milestone and required consumer.

## Durability vocabulary used in this table

- `durable_uri` — an independent, off-host location from which the artifact can be *fetched*.
  `NONE` means no such location has been confirmed to exist.
- `recoverability` — what a fresh clone can do with the artifact today:
  - `VERIFY-ONLY` — the artifact cannot be fetched from anywhere off-host. A copy obtained
    out-of-band (from the host named in `local_path`) can be integrity-checked against the
    recorded SHA256, but the artifact itself must be transported manually.

## What a reviewer can do from a fresh clone

| | |
| --- | --- |
| **In-repo** | all source; the frozen evaluation protocol, cohort definition and preregistrations; the frozen reports; and the SHA256 of every accepted artifact in the table below |
| **Verifiable from the clone alone** | that each report matches the hash recorded for it, and that the recorded protocol and source hashes are unchanged |
| **Not possible from the clone alone** | recomputing the reported benchmark numbers, or serving the real-data demo — both need the large artifacts below, which are not distributed from this repository |
| **Still available offline** | the synthetic quick start (`./scripts/run_demo.sh`) and the whole test suite: no downloaded artifact, no credential and no network call |

The synthetic path makes the pipeline, the trust boundaries and the HTTP/demo contract runnable. It
is not evidence about recommendation quality, and none of the reported numbers comes from it.

## Corrected artifact location record

The locations below were verified by direct measurement during the M1 durability audit
(`sha256sum` computed on the host holding each file). See the CORRECTION NOTE at the end of this
document: the `runs/...` paths this table previously carried were wrong as storage locations.

| artifact (repo-relative name) | local_path (verified present) | host | size (B) | SHA256 | durable_uri | recoverability |
|---|---|---|---|---|---|---|
| `runs/tiger_public_2026/sid/semantic_ids.json` | `/root/autodl-tmp/agentrec-X/runs/tiger_public_2026/sid/semantic_ids.json` | AutoDL | 5775237 | `6501fbe8e146e3c57ce7a146e37c3be29235ae5bbd082116f6a8e30e739e8d0e` | NONE | VERIFY-ONLY |
| `runs/tiger_public_2026/sid/layout.json` | `/root/autodl-tmp/agentrec-X/runs/tiger_public_2026/sid/layout.json` | AutoDL | 394 | `22521d7b58f8538c3d3b0d8a27200a255ac62e1e9da0676c6c55ebadde101b11` | NONE | VERIFY-ONLY |
| `runs/tiger_public_2026/sid/tokenizer.pt` | `/root/autodl-tmp/agentrec-X/runs/tiger_public_2026/sid/tokenizer.pt` | AutoDL | 6853270 | `d514aafcddf0edb922a18f463520c2bc80b7a0a8eb46e358e4fa084642c610c3` | NONE | VERIFY-ONLY |
| `runs/gate_d_fp32_canonical_remediation/tiger.pt` | `/root/autodl-tmp/gate_d_fp32_canonical_remediation/tiger.pt` | AutoDL | 45181927 | `95e5cb6f816bb271b8551a6dd9b62ee33c8e0f6b5908ce426622113a072c6c0a` | NONE | VERIFY-ONLY |
| `runs/gate_d_fp32_canonical_remediation/generator_layout.json` | `/root/autodl-tmp/gate_d_fp32_canonical_remediation/generator_layout.json` | AutoDL | 1267 | `eed4dffaccf8dfecf40ee5bc9f8206870910a8a4e31fdd941e8ac6697814a55a` | NONE | VERIFY-ONLY |
| `runs/gate_d_fp32_canonical_remediation/tiger.json` | `/root/autodl-tmp/gate_d_fp32_canonical_remediation/tiger.json` | AutoDL | 7871 | `5d47bfd6a51635a74b7552daaec2a62f7092db5b874031e029fe834d0d280cc6` | NONE | VERIFY-ONLY |
| `runs/step26_h7_canonical/retrieval_candidates.jsonl` | `/root/autodl-tmp/step26_h7_canonical/retrieval_candidates.jsonl` | AutoDL | 257443487 | `6cb5b0870c81d9a45524a536fb82560cef7e0e77284f56256d1d001e3f59c5f2` | NONE | VERIFY-ONLY |
| `runs/step26_h7_canonical/retrieval_summary.json` | `/root/autodl-tmp/step26_h7_canonical/retrieval_summary.json` | AutoDL | 2896 | `a48974ba381132031077b98c61de1c621937689b5cf550b032cc3ec3c137a79b` | NONE | VERIFY-ONLY |
| `runs/sasrec_canonical_2026/best.pt` | `/mnt/d/IT/CODE/PROJECTION/agentrec/X/runs/sasrec_canonical_2026/best.pt` | developer workspace (WSL) | 121675470 | `352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912` | NONE | VERIFY-ONLY |
| `runs/twotower_public_2026_full/checkpoint.pt` | `/mnt/d/IT/CODE/PROJECTION/agentrec/X/runs/twotower_public_2026_full/checkpoint.pt` | developer workspace (WSL) | 162221222 | `0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab` | NONE | VERIFY-ONLY |
| `runs/semantic_id_public_2026/generator.pt` | `/mnt/d/IT/CODE/PROJECTION/agentrec/X/runs/semantic_id_public_2026/generator.pt` | developer workspace (WSL) | 562044 | `1d7d5a5f44a23e074db3d210ab4f6b0a33e2344427d80e196752d323e4d41b14` | NONE | VERIFY-ONLY |

All eleven artifacts are `durable_uri: NONE` / `recoverability: VERIFY-ONLY`. No off-host copy is
confirmed for any of them; none is in Git history and none is present in a fresh clone.

Two SHA256 values were not recorded in M1 (`runs/sasrec_canonical_2026/best.pt`,
`runs/semantic_id_public_2026/generator.pt`). They were measured in this audit and are recorded
above for the first time; they are first measurements, not re-verifications of an earlier value.

## Split across two private hosts

The eleven artifacts are not even co-located: eight exist only on the AutoDL host and three exist
only in the developer workspace. The AutoDL copies of `tiger.pt` exist in four directories
(`gate_d_canonical_r1`, `gate_d_failed_run_6447d75_20260923`, `gate_d_remediation_smoke_out`,
`gate_d_fp32_canonical_remediation`), but all four are on the same physical volume
(`/dev/md0`), so they are redundant against file loss, not against host or volume loss.

## How a fresh clone obtains them

1. It cannot. A fresh clone contains neither the artifacts nor a fetchable pointer to them
   (`durable_uri: NONE` for every row above).
2. What a fresh clone *can* do is integrity-check a copy it obtains out-of-band:
   `sha256sum <path>` against the table above.
3. The FP32 canonical checkpoint and the H7 certified retrieval artifact are regenerable, in
   principle, by the documented pipeline rather than by M1: see
   `docs/reports/step26-h7-execution-manifest.txt` for the exact H7 command and its frozen input
   hashes, and `docs/reports/fp32-canonical-remediation-report.txt` for the training provenance.
   Regeneration is not a durability guarantee here, because every frozen input named by those
   documents (the FP32 checkpoint directory, the SID directory, the 161 MB backend handoff
   catalogue) is itself host-local on the AutoDL machine.
4. **Known gap (unchanged, now measured precisely):** no public object store and no release asset
   hosts these artifacts, and no cloud transfer tooling (`rclone`, `ossutil`, `aws`, `gsutil`) is
   configured on either host. Uploading them to durable external storage remains an open item
   (see `docs/PROJECT_STATE.md` known deferred work).

## B0 recoverability promotion — canonical manifest view

`config/demo_runtime_artifacts.json` is the **canonical machine-readable manifest**. This section is
its human-readable view for the artifacts whose durable restore is being established (B0 scope).
The M1 audit table above is unchanged and remains the historical record of the eleven artifacts;
the two views answer different questions and neither replaces the other.

| role | path | size (B) | SHA256 | durable status | required_for |
| --- | --- | --- | --- | --- | --- |
| `two_tower_checkpoint` | `runs/twotower_public_2026_full/checkpoint.pt` | 162221222 | `0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab` | `READY` | `two_tower_serving_smoke` |
| `two_tower_metadata_categories` | `runs/twotower_public_2026_full/metadata_categories.json` | 2741318 | `46469955655b8c552a172cb324e724455c72ae4bc10c9fe3d53a35c83b882a9d` | `READY` | `two_tower_serving_smoke` |
| `sasrec_checkpoint` | `runs/sasrec_canonical_2026/best.pt` | 121675470 | `352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912` | `READY` | `serving_fusion_v1` |
| `mappings` | `data/processed/Sports_and_Outdoors_mappings.json` | 39670810 | `dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912` | `READY` | `two_tower_serving_smoke`, `demo_runtime` |
| `catalogue` | `data/processed/Sports_and_Outdoors_products.jsonl` | 307500147 | `175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd` | `READY` | `serving_fusion_v1` |
| `sequences` | `data/processed/Sports_and_Outdoors_sequences.json` | 348683134 | `d3d834260c867ff39ac1b32bb08fd17ea1dc394e9a6fb7af8d19aa046c1eab81` | `READY` | `demo_runtime` |

**Status today: `READY` — a clean restore has been proven (B0-c).** The three artifacts were
fetched over the network into a workspace that contained **no local copy** of any of them, their
sizes and SHA-256 were re-verified, `TwoTowerInferenceEngine` loaded, and `recommend(k=10)` returned
10 legal items with `PAD` excluded and every `parent_asin` round-tripping to its `item_id`. The gap
recorded in "How a fresh clone obtains them" above is therefore closed **for these three artifacts
only**; the rest of that gap (SASRec checkpoint, catalogue, sequences) is untouched.

**The scope grew by one file, and the restore is what revealed it.** B0-b uploaded two artifacts.
At load time the engine refused to construct:

```text
the checkpoint was trained with categorical metadata but the category artifact is missing:
runs/twotower_public_2026_full/metadata_categories.json
```

`TwoTowerInferenceEngine` resolves that file as a **sibling of the checkpoint**, so the checkpoint is
not usable without it. `two_tower_metadata_categories` was added to the scope, uploaded, and included
in the restore. It is recorded in the manifest with a `scope_note` saying why.

### Remote locations (B0-b, verified by B0-c)

| role | repo_id | revision | remote path | commit |
| --- | --- | --- | --- | --- |
| `two_tower_checkpoint` | `autorec/agent-rec-x-artifacts` | `main` | `serving/twotower/checkpoint.pt` | `794f5d901804` |
| `mappings` | `autorec/agent-rec-x-artifacts` | `main` | `serving/catalog/Sports_and_Outdoors_mappings.json` | `647c65f2fe7c` |
| `two_tower_metadata_categories` | `autorec/agent-rec-x-artifacts` | `main` | `serving/twotower/metadata_categories.json` | `668a2d44b711` |

The repository is **private**, so both a reachable network and a read credential are required.
Upload and restore share one convention: `repo_type` is frozen in the manifest as `dataset`, and
the collection segment (`datasets/`) is *derived* from it rather than recorded in `repo_id`.

### Durable status lifecycle

```text
PENDING                no remote location exists yet; fetch refuses and never substitutes a local copy
UPLOADED_UNVERIFIED    uploaded and remotely queryable (path, size and content sha256 verified),
                       but NO clean restore has been proven; fetch may run -- that is how B0-c works
READY                  a clean restore has been proven; only a passing B0-c may set this
```

An upload alone never sets `READY`; each `READY` entry carries `verified_at`, `restore_sha256` (the
digest computed on the freshly restored file) and `smoke_result` in the manifest.

The backend is Hugging Face and the download location is **derived, never stored**. The repository
kind — frozen in the manifest as `recoverability.repo_type`, one of `model` or `dataset` — selects
the path shape, and an unrecognised kind is refused rather than guessed:

```text
model repo    <endpoint>/<repo_id>/resolve/<revision>/<remote_path>
dataset repo  <endpoint>/datasets/<repo_id>/resolve/<revision>/<remote_path>

endpoint   = https://huggingface.co          (manifest: recoverability.endpoint)
repo_type  = dataset                         (manifest: recoverability.repo_type)
repo_id    = $HF_REPO_ID                     (or durable.repo_id in the manifest)
revision   = $HF_REVISION                    (default: main)
```

`repo_id` is always the bare `owner/name`: the collection segment is derived from `repo_type`, so a
`repo_id` that already carries `datasets/` or `models/` is **rejected** — accepting it would produce
`/datasets/datasets/...` and a 404 that looks like a permissions problem.

Because the URI is derived from its parts, a recorded address cannot drift away from the recorded
repository. A private or gated repository needs `HF_TOKEN`; a public one does not.

### Configuration precedence

```text
real process environment  >  .env.local  >  .env  >  manifest default
```

A variable set in the real environment is never overridden by a file, and an **empty value counts
as unset** rather than as a deliberate empty string. `HF_TOKEN` is sent only as a request header,
is never printed, never written to the manifest and never included in an error message; it is also
dropped when a redirect leaves the original host.

### Restore and verification procedure

```bash
python scripts/fetch_artifacts.py list      # identity, derived URI, status; offline
python scripts/fetch_artifacts.py verify    # size + SHA256 of local copies; offline
python scripts/fetch_artifacts.py probe     # lightweight network validation; network, no bodies
python scripts/fetch_artifacts.py fetch --all   # download + verify; network
python scripts/fetch_artifacts.py smoke     # fresh-clone contract; network
```

`probe` answers "is the durable side reachable and correct?" **without transferring an artifact**:

| check | request | answers |
| --- | --- | --- |
| A. repo access | `GET /api/<models\|datasets>/<repo_id>` | does the repository exist and may this token read it? |
| B. revision/tree | `GET /api/<models\|datasets>/<repo_id>/tree/<revision>` | does the revision exist? |
| C. artifact path | `HEAD <derived resolve URI>` | does the remote path exist, and what size does the service report? |

Check C sends a `HEAD` and deliberately does **not** follow redirects, so the service's own size
hint (`x-linked-size`, else `Content-Length`) is readable and no artifact body can be pulled. That
size is compared with the manifest's `size_bytes`; if the service provides neither header the probe
reports **size unavailable** — neither a pass nor a failure, because the manifest cannot be checked
from that response.

A probe failure is reported by cause, so one cause is never mistaken for another:

```text
connection-failure / timeout  -> network egress or firewall; NOT a credential problem
authentication/permission     -> 401/403: token validity, scope, or repository visibility
not-found                     -> 404: repo_id spelling/case, repo_type, revision, or remote_path
http-error                    -> any other status, reported with its code
skipped                       -> nothing to test yet (no repo_id, or no remote_path recorded)
```

Exit codes: `0` success, `2` blocked (not configured, or `PENDING`), `3` verification failure,
`4` transport failure. A failed download is written to `<path>.part` and deleted on any size or
digest mismatch, so a partial transfer is never installed as an artifact.

### B1 evidence durability (B0-b)

The seven large B1 evidence binaries — the depth-1000 Two-Tower head, the four per-arm hit files,
the paired inputs and the bootstrap draws — were uploaded alongside the two runtime artifacts, so
that the evidence behind the B1 paired result no longer exists only on the development machine.
Their remote prefix is `evidence/b1/<original filename>`.

This is **storage metadata and nothing else.** It does not alter the B1 statistical result, the
execution manifest or the frozen preregistration, and it cannot: the record
[`runs/b1_evidence/B1_DURABILITY.json`](../runs/b1_evidence/B1_DURABILITY.json) states that
explicitly, lists each file's local path, remote path, size, sha256, repo, revision and upload
status, and defers the verdict to `B1_EXECUTION_MANIFEST.json`.

Like the runtime artifacts, these files are `UPLOADED_UNVERIFIED`: their content sha256 is
confirmed by the repository's own object id, but no restore has been performed.

### The fresh-clone smoke contract

A `smoke` PASS means all of the following, in order:

```text
clean workspace -> target artifact is ABSENT locally
                -> fetched over the network from the durable URI
                -> SHA256 re-verified against the canonical manifest
                -> TwoTowerInferenceEngine loads (checkpoint + canonical mappings)
                -> recommend(k=10) returns 10 items
                -> PAD (item_id 0) is not among them
                -> every returned parent_asin round-trips to its item_id
```

The first line is enforced as a refusal: if the target files already exist, `smoke` stops and says
so. **Copying an artifact from a development machine's own directory is not recoverability
evidence**, so a run that never crossed the network cannot pass this contract.

Credentials live in `.env` / `.env.local` (git-ignored; see `.env.example`). A token is never
written to the manifest, to this document, to a test fixture or to any log, and it is dropped when a
redirect leaves the original host.

## CORRECTION NOTE — artifact paths were recorded against the wrong root

This document previously presented each artifact under a repository-relative `runs/...` path, in a
table headed "canonical path", and the "Current storage location" section asserted that those
paths are "relative to the AutoDL working host (`/root/autodl-tmp/agentrec-X`)". Both statements
were wrong as statements about where the files are.

Verified during the M1 durability audit:

- `runs/` in this repository is gitignored, so no artifact was ever a repository path.
- On the AutoDL host, `/root/autodl-tmp/agentrec-X/runs/` contains only four entries
  (`demo_catalog`, `step26_tiger_benchmark`, `tiger_backend_handoff`, `tiger_public_2026`).
  Eight of the eleven artifacts are **not** under it; they live at the top level of the data
  volume, i.e. `/root/autodl-tmp/<name>/...` rather than `/root/autodl-tmp/agentrec-X/runs/<name>/...`.
- The remaining three (`best.pt`, `checkpoint.pt`, `generator.pt`) are absent from the AutoDL host
  entirely and exist only in the developer workspace at
  `/mnt/d/IT/CODE/PROJECTION/agentrec/X/runs/...`.

Effect on the recorded hashes: none. All nine SHA256 values that M1 recorded were re-verified
against the real files during this audit and all nine MATCH, including the two large ones
(`tiger.pt` 45181927 B and `retrieval_candidates.jsonl` 257443487 B). No artifact was moved,
renamed, copied or regenerated by this correction; only this document changed.

Scope: this correction records where the artifacts are and what a fresh clone can do with them. It
does not create durable hosting, and it does not close the M1 artifact-durability gap.
