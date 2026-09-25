# Large-artifact storage and verification

M1 deliberately does NOT commit large artifacts. The repository holds code, tests, experiment
scripts, configuration, small manifests and accepted reports only. Every large artifact below is
recorded by canonical path, SHA256, producer milestone and required consumer.

| canonical path | what | SHA256 | producer | consumer |
|---|---|---|---|---|
| `runs/tiger_public_2026/sid/semantic_ids.json` | Semantic-ID assignment (156746 items) | 6501fbe8e146e3c57ce7a146e37c3be29235ae5bbd082116f6a8e30e739e8d0e | Step 2.4F | reproducibility of the milestones that cite it |
| `runs/tiger_public_2026/sid/layout.json` | accepted SID token layout v3 | 22521d7b58f8538c3d3b0d8a27200a255ac62e1e9da0676c6c55ebadde101b11 | Step 2.4F | reproducibility of the milestones that cite it |
| `runs/tiger_public_2026/sid/tokenizer.pt` | RQ-VAE tokenizer weights | d514aafcddf0edb922a18f463520c2bc80b7a0a8eb46e358e4fa084642c610c3 | Step 2.4F | reproducibility of the milestones that cite it |
| `runs/gate_d_fp32_canonical_remediation/tiger.pt` | accepted FP32 TIGER generator checkpoint (45181927 B) | 95e5cb6f816bb271b8551a6dd9b62ee33c8e0f6b5908ce426622113a072c6c0a | Gate D / FP32 remediation | reproducibility of the milestones that cite it |
| `runs/gate_d_fp32_canonical_remediation/generator_layout.json` | derived generator layout (vocab 1028) | eed4dffaccf8dfecf40ee5bc9f8206870910a8a4e31fdd941e8ac6697814a55a | R1-A | reproducibility of the milestones that cite it |
| `runs/gate_d_fp32_canonical_remediation/tiger.json` | training metadata for the accepted checkpoint | 5d47bfd6a51635a74b7552daaec2a62f7092db5b874031e029fe834d0d280cc6 | Gate D | reproducibility of the milestones that cite it |
| `runs/step26_h7_canonical/retrieval_candidates.jsonl` | H7 certified retrieval, 20000 cases (257443487 B) | 6cb5b0870c81d9a45524a536fb82560cef7e0e77284f56256d1d001e3f59c5f2 | Step 2.6 H7 | reproducibility of the milestones that cite it |
| `runs/step26_h7_canonical/retrieval_summary.json` | H7 certificate + provenance | a48974ba381132031077b98c61de1c621937689b5cf550b032cc3ec3c137a79b | Step 2.6 H7 | reproducibility of the milestones that cite it |
| `runs/sasrec_canonical_2026/best.pt` | accepted SASRec checkpoint | not hashed in M1 | Phase 1 | reproducibility of the milestones that cite it |
| `runs/twotower_public_2026_full/checkpoint.pt` | accepted full-exposure Two-Tower checkpoint | 0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab | Phase 5 / model expansion | reproducibility of the milestones that cite it |
| `runs/semantic_id_public_2026/generator.pt` | frozen GenRec-v0 generator | not hashed in M1 | Phase 4 historical | reproducibility of the milestones that cite it |

## Current storage location

These paths are relative to the **AutoDL working host** (`/root/autodl-tmp/agentrec-X`) and to a
developer workspace that mirrors `runs/`. They are NOT in Git history.

## How a fresh clone obtains them

1. The FP32 canonical checkpoint and the H7 certified retrieval artifact are produced by the
   documented pipeline, not by M1: see `docs/reports/step26-h7-execution-manifest.txt` for the
   exact H7 command and its frozen input hashes, and
   `docs/reports/fp32-canonical-remediation-report.txt` for the training provenance.
2. Verify any artifact with `sha256sum <path>` against the table above before using it.
3. **Known gap:** there is currently no public object store or release asset hosting these
   artifacts. They live on one host. M1 records their hashes so a future copy is verifiable, but
   a fresh clone cannot yet *fetch* them. Uploading them to durable external storage is an open
   item (see `docs/PROJECT_STATE.md` known deferred work).

