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
