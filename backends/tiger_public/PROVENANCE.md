# Provenance and licence status

Every file in this directory is labelled with one of three origins. `T6` in
`tests/test_backend_boundaries.py` asserts that this file labels **every** file under
`backends/tiger_public/`, so a vendored file cannot be added unnoticed.

Labels:

* **`ORIGINAL`** — written for AgentRec-X.
* **`REIMPLEMENTED_FROM(<repo>@<sha>, <file>:<lines>)`** — the algorithm was re-derived from a
  cited source. No verbatim code was copied.
* **`DEPENDENCY(<package>==<version>, <licence>)`** — an external package or model checkpoint,
  used as a dependency rather than copied.

## Why nothing is vendored

Both audited repositories are legally *read-only design sources*.

| Repository | Licence found | Consequence for this project |
| --- | --- | --- |
| [`mclwu22/amazon-genrec`](https://github.com/mclwu22/amazon-genrec) | **no `LICENSE` file** — no licence is granted, so the default is all rights reserved | its code is **not** copied. Its *design* is re-derived and cited below. |
| [`snap-research/GRID`](https://github.com/snap-research/GRID) | **Snap Inc., "non-commercial, research purposes only"**, with a mandatory notice-retention clause on redistribution | its code is **not** vendored. Its *design* is re-derived and cited below. |

If an excerpt from GRID is ever quoted in this tree, the Snap notice must be reproduced
verbatim alongside it. None is quoted today.

## Per-file status

### `pyproject.toml`
`ORIGINAL`. Declares `DEPENDENCY(numpy<2, BSD-3-Clause)` for Step 2.3. Steps 2.4-2.6 will add
`DEPENDENCY(torch, BSD-3-Clause)`, `DEPENDENCY(transformers, Apache-2.0)` and
`DEPENDENCY(sentence-transformers, Apache-2.0)` to **this** venv only.

### `README.md`, `PROVENANCE.md`, `.gitignore`
`ORIGINAL`.

### `src/tiger_public/__init__.py`
`ORIGINAL`. Package docstring and the `STUB_MARKER` constant.

### `src/tiger_public/contracts.py`
`ORIGINAL` for the schemas, the PAD sentinel, the token layout and the guards.
Portions are `REIMPLEMENTED_FROM` the following, with the changes stated:

* `snap-research/GRID@2fe3475b2d369580234093f35d52b1a2f54d0472`,
  `src/utils/tensor_utils.py:125-176` — the collision-deduplication *algorithm*
  (`deduplicate_rows_in_tensor`). **Changed:** the first member of each collision group is
  assigned `0` rather than `1` (GRID assigns `1..k` to every member); the ordinal pass is a
  single `O(N)` scan of a dictionary rather than an `O(groups × N)` `torch.where` loop; the
  `.pt`-only file guard and the in-place read/write wrapper are not adopted; and an explicit
  **overflow refusal** is added, which the source does not have.
* `snap-research/GRID@2fe3475b2d369580234093f35d52b1a2f54d0472`,
  `src/models/modules/clustering/base_clustering_module.py` and
  `src/components/loss_functions.py` — the commitment coefficient `0.25` and the
  squared-Euclidean distance convention. **Changed:** only the constants and the convention are
  adopted; the Lightning module that hosts them is not.

### `src/tiger_public/io.py`
`ORIGINAL`. Manifest hashing, the forbidden-key refusal, and the handoff readers. The
"no target is visible to the backend" rule is enforced here mechanically rather than by
convention.

### `src/tiger_public/scoring.py`
`ORIGINAL`. The frozen item-score rule
(`score(i | history) = Σ_l log P_full-vocabulary(t_l | history, t_<l)`, EOS excluded) and the
prefix-bound admissibility helper.

The rule itself follows the *definition* used by
`mclwu22/amazon-genrec` (`tiger/50_train.py`, which scores generated items by the model's own
sequence log-probability) and by `snap-research/GRID`
(`src/models/modules/semantic_id/tiger_generation_model.py`, `_beam_search_one_step`).
**Changed:** the rule is stated once, separately from any search algorithm, and child
renormalisation is explicitly forbidden because it would make a score depend on the search
frontier. Neither source states the rule independently of its decoder.

### `src/tiger_public/features.py`
`ORIGINAL` for the streaming encoder driver, the atomic artifact write, the corruption guards
and the deterministic smoke stand-in.
Portions are `REIMPLEMENTED_FROM`:

* `mclwu22/amazon-genrec`, `tiger/20_embed.py:21-57` — the frozen sentence encoder stage:
  one row per item, an aligned id sidecar, `normalize_embeddings=False`, a `max_chars` cap and
  batched encoding with progress. **Changed:** the text is read from AgentRec-X's
  `products_text.jsonl` rather than composed here, the ids are opaque integers rather than
  canonical strings, the array is written to a temporary path and renamed only on success, rows
  are streamed into a memory-mapped file instead of being accumulated, and non-finite rows are
  replaced with the batch mean and counted rather than written through.

### `src/tiger_public/quantizer.py`
`ORIGINAL` for the training loop, the dead-code revival, the deterministic seeding and the
diagnostics wrapper.
Portions are `REIMPLEMENTED_FROM`:

* `snap-research/GRID@2fe3475b2d369580234093f35d52b1a2f54d0472`,
  `src/modules/clustering/residual_quantization.py`, `src/modules/clustering/vector_quantization.py`,
  `src/components/{quantization_strategies,distance_functions,loss_functions,clustering_initializers}.py`
  — the encoder/decoder geometry (*hidden* widths `[768, 256, 128]` with the latent layer
  appended by the model, i.e. `768 -> 256 -> 128 -> 64`), input normalisation, squared-Euclidean
  nearest-code search, the straight-through estimator, `BetaQuantizationLoss` with
  `beta = 0.25`, k-means++ codebook seeding with the degenerate-case escape, and the per-level
  coverage / `id_entropy` diagnostics. **Changed:** the Lightning module, the DDP rank-0
  broadcast protocol, the `MiniBatchKMeans` wrapper, the Hydra `_target_` wiring and the
  `eval_step` defect are not adopted; the loop is plain PyTorch, the initialisation runs in one
  process, and the dead-code revival follows amazon-genrec rather than GRID (GRID only
  *reports* collapse).
* `mclwu22/amazon-genrec`, `tiger/30_rqvae.py:73-84` — dead-code revival: unused codes are
  re-seeded from sampled data each epoch. **Changed:** applied to encoder latents under the
  model's own generator, with the revived count reported per epoch.
* `snap-research/GRID@2fe3475b2d369580234093f35d52b1a2f54d0472`,
  `src/modules/clustering/base_clustering_module.py` — the measured finding that Adam at
  `learning_rate = 1e-3` collapses the later residual levels; see the Step-2.4 report.

### `src/tiger_public/generator_layout.py`
`ORIGINAL`.  The generator's token vocabulary as a **derived** artifact rather than a patched copy
of the accepted SID layout.  It reads `sid/layout.json` read-only, copies every level, offset,
codebook value and the pad/bos/eos ids verbatim, appends `sep` above the accepted vocabulary, and
records `source_sid_layout_sha256` + `source_semantic_ids_sha256` so the derivation is auditable.
It refuses a special inside the catalogue code space, a non-distinct special, or a special that
aliases a token the accepted Semantic IDs actually use.  No external source: the accepted artifact
is this project's own, so nothing here is absorbed from either reference repository.

### `src/tiger_public/trie.py`
`ORIGINAL` for the trie, its stats and its fail-closed resolution.
`REIMPLEMENTED_FROM` `mclwu22/amazon-genrec`, `tiger/50_train.py:36-49` (`build_trie`) and
`snap-research/GRID@2fe3475b`, `src/models/modules/semantic_id/tiger_generation_model.py:202-251`
(`_check_valid_prefix`). **Changed:** the trie is built from the accepted catalogue assignment
rather than a separate validated-token file, so it cannot drift from the SIDs actually in use;
construction *refuses* a layout whose sentinel is tokenisable; the dedup level is its own token
block via the layout's offsets; and the structure is a real prefix trie (O(1) child lookup)
rather than GRID's `O(C x b x H)` membership test, which its own TODO notes is a placeholder.

### `src/tiger_public/tiger.py`
`ORIGINAL` for the example builder, the local trainer, the constrained beam search and the
dependency-bound checkpoint I/O.
Portions are `REIMPLEMENTED_FROM`:

* `mclwu22/amazon-genrec`, `tiger/50_train.py` — the seq2seq direction: a randomly initialised
  T5 encoder-decoder over SID tokens, one item block plus EOS as the target, AdamW with a
  OneCycleLR schedule, bf16 autocast on a CUDA host, and constrained beam search over a
  catalogue trie. **Changed:** training reads AgentRec-X's `train_exposure.jsonl` instead of
  self-derived token files; token ids come from the accepted `layout.json` offsets rather than
  hardcoded `VOCAB = 1027` constants repeated across files; the trainer is plain PyTorch with no
  Lightning/Hydra/DDP; padding is masked and never a target; and the checkpoint is bound to
  dependency hashes, which the source does not do.
* `snap-research/GRID@2fe3475b`, `src/models/modules/semantic_id/tiger_generation_model.py` —
  the per-level offset idea (one embedding table serving several codebooks via disjoint offset
  blocks) and the beam-search shape. **Changed:** the offsets are derived from the accepted
  layout, not from a `codebook_size` argument; decoding consults a prefix trie so an illegal
  continuation cannot be produced at all, rather than being filtered afterwards; and the shipped
  GRID prefix check is disabled by default, which this implementation does not inherit.

### `src/tiger_public/dedup.py`
`ORIGINAL` for the module boundary, the overflow refusal and the recorded audit shape.
The collision-ordinal scheme is `REIMPLEMENTED_FROM`
`snap-research/GRID@2fe3475b2d369580234093f35d52b1a2f54d0472`,
`src/utils/tensor_utils.py:125-176` (`deduplicate_rows_in_tensor`) and
`mclwu22/amazon-genrec`, `tiger/30_rqvae.py:165-170`. **Changed:** the first member of each
group is assigned `0` rather than `1`; the pass is one `O(N)` scan instead of an
`O(groups x N)` loop; a zero-row (PAD) assignment is refused rather than silently numbered; and
an oversize group raises instead of wrapping or sharing a final identity — neither source has
that refusal.

### `src/tiger_public/cli.py`
`ORIGINAL` for the stage wiring, the placeholder generators, the manifest writing and the
streaming protocol.
Portions are `REIMPLEMENTED_FROM`:

* `mclwu22/amazon-genrec`, `tiger/40_make_tokens.py:24-30` — the per-level token-offset layout
  (`level_offsets = [0, K, 2K, ...]`) and item-major flattening. **Changed:** the offsets and
  the special tokens are *derived* from the layout rather than repeated as the magic constant
  `VOCAB = 1027` in three files, and specials are placed strictly above the code space rather
  than at a fixed numeric value.
* `mclwu22/amazon-genrec`, `tiger/50_train.py:91-97` — holding batches resident on the device
  and slicing a permutation (its comment records a DataLoader-bound ~3 steps/s). **Not adopted
  in Step 2.3**, which trains nothing; recorded here because Step 2.5 will reuse the pattern.
* `snap-research/GRID@2fe3475b2d369580234093f35d52b1a2f54d0472`,
  `src/modules/clustering/residual_quantization.py:501-581` — the per-layer coverage and
  `id_entropy` diagnostics. **Changed:** the coverage/entropy pair is computed inline and the
  `>20%` dead-code rule is a *catastrophic-collapse refusal*, not a comparative quality gate.

### `REPRODUCIBILITY.md`, `requirements-ml.txt`
`ORIGINAL`.  The pinned ML runtime and the two separate decisions it records: package versions
(frozen to what Step 2.4 actually exercised) and the CUDA wheel/index (deliberately chosen on
the production host, because a wheel selected here would be a guess about that host's driver).
`DEPENDENCY(numpy==1.26.4, BSD-3-Clause)`,
`DEPENDENCY(transformers==5.17.0, Apache-2.0)`,
`DEPENDENCY(sentence-transformers==6.1.0, Apache-2.0)`,
`DEPENDENCY(pyarrow, Apache-2.0)`, `DEPENDENCY(pytest, MIT)`.

### `tests/conftest.py`, `tests/__init__.py`, `tests/test_backend_local.py`
### `tests/test_step24_features.py`, `tests/test_step24_quantizer.py`
`ORIGINAL`.  The Step-2.4 test modules exercise the encoding stage, the RQ-VAE arithmetic, the
k-means++ seeding, the straight-through gradient path, the dedup ordinals and the overflow
refusal.  They need no model weights, because the smoke encoder is deterministic and offline.

### `tests/test_step25_tiger.py`
`ORIGINAL`.  Generator and trie tests: example construction from train history only, PAD
exclusion, dedup-level token separation, trie completeness and invalid-prefix rejection,
checkpoint dependency-mismatch refusal, and a small train/save/load/constrained-generate smoke.
Example provenance is asserted **positionally** - the source is exactly
`train_history[max(0, k - max_hist_items):k]` with no position at or after `k` - and a
repeat-purchase case (`[A, B, A] -> input [A, B], target A`) is retained as valid, because an
item-set disjointness test would wrongly delete every repurchase.  The Gate-C registration is
pinned against both the dataclass and the CLI defaults, so a drifting default fails here instead
of after a GPU run.  Model-building tests use a deliberately tiny architecture because the
registered production architecture is *reported*, not enforced; a test that could not afford
14 M parameters would skip the very code paths that need testing.

### `experiments/smoke_tiger_step25.py`, `experiments/gate_d_tiger_training.py`
`ORIGINAL`, and outside this package's licence-sensitive tree (they live in AgentRec-X's
`experiments/`).  The Gate-B smoke exercises the generator end to end on a tiny CPU model; the
Gate-D runbook check restores and verifies the accepted Step-2.4F archive, streams the frozen
exposure arithmetic, and refuses to print a training command unless every hash and count matches.
Neither rebuilds features, the RQ-VAE or the Semantic IDs.  `gate_d_tiger_training.py` is the only
place that rewrites an accepted artifact, and only under the explicit `--patch-layout` flag
(additive `sep`/`vocab_size` metadata, with a backup and no change to any Semantic ID).

### `tests/test_step24f_readiness.py`, `tests/test_step24f_audit.py`
`ORIGINAL`.  The Step-2.4F test modules cover encoder-revision pinning and the refusal to record
an unconfirmed pin, config reconstruction from a stored artifact block, and the read-only audit's
arithmetic: prefix reconstruction MSE, residual norms, complete occupancy histograms, the
largest-code fraction as distinct from codebook coverage, and the pre-dedup collision-group
distribution.  Neither module needs network access.

## Corrections and notices carried with the backend

These are recorded because source-fidelity matters for a licence-sensitive tree.  They are
notices, not vendored code.

1. **The smoke encoder's User-Agent default.** The stand-in in `features.py` is described as
   character-ngram hashing.  One detail of that description was inherited from an earlier draft
   and does not match the code: no User-Agent string is set anywhere in this tree, and
   `features.py` performs no HTTP request of its own, so it never sends one.  Network access, if
   any, happens inside `transformers`/`huggingface_hub` when loading weights.
2. **Host-specific configuration is not baked in.** No mirror endpoint is hardcoded;
   `HF_ENDPOINT`, `HF_HOME` and `HF_HUB_CACHE` are read from the environment only.  A production
   run sets them explicitly, and the values used belong in the run log rather than in the code.
3. **The encoder revision is a snapshot SHA-1, not a tag.** `fc5d4628...` was recovered from the
   Hub cache that produced the Step-2.4 artifact.  It is recorded as the pin because a tag or a
   repository name is mutable and would not support the reproducibility claim.

## What is deliberately **not** absorbed

From either repository: every evaluator, every metric, every split, every dataloader, every
launcher, and every restart/DDP/BigQuery/TFRecord module. The backend owns none of those, and
`T4` in the AgentRec-X suite asserts that no module named for one exists here.

Specifically not adopted:

* `amazon-genrec`'s Spark pipeline, its `spark/60_split.py` leave-two-out split, and its
  beam-limited `70_eval.py` metric harness;
* `GRID`'s Lightning/Hydra config tree, its TFRecord data layer, its `SIDRetrievalEvaluator`
  (which scores a collision as a hit and applies no seen-item masking), its
  `should_check_prefix` decoder constraint (implemented but never enabled by any config, and a
  brute-force membership test rather than a trie), and its known `eval_step` unpack defect.
