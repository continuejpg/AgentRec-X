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

### `tests/conftest.py`, `tests/__init__.py`, `tests/test_backend_local.py`
`ORIGINAL`.

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
