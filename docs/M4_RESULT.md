# M4 FORMAL RESULT — fixed four-source traversal vs LLM adaptive source selection

**Status:** FROZEN. The run is complete, executed exactly once, and was not re-run.

Every number below is transcribed from the single result artifact
`runs/m4_formal_result.json`. Nothing here was recomputed, re-derived or repaired. The artifact is
gitignored under AGENTS.md §8 (large generated experiment output), which is precisely why its
contents are recorded in this committed document: without this file the formal result would exist
only on one machine.

Governing documents: **original preregistration + Amendment 1**.

---

## 1. Provenance

**Two different commits matter, and they are not the same commit.**

| role | identity |
|---|---|
| **executed code commit** — the code the formal experiment actually ran | `0e6a7e82f8a03f6081c62ca317df9a6e95bb3055` ("M4: provider validation, concurrency probe, and parallel arm execution") |
| **archive / seal commit** — the commit that froze this record | the annotated Git tag **`M4-sealed`**; resolve with `git rev-list -n1 M4-sealed` |

The execution manifest names the first as `code.executed_code_commit` (renamed from the ambiguous
`code.commit`) and names the second only as the ref `code.seal_ref`. A manifest **cannot** record
the hash of the commit that contains it: writing the manifest changes the working tree, so recording
the seal commit changes the seal commit. The seal is therefore expressed as a Git ref, which Git
resolves authoritatively, and the manifest deliberately embeds no self-referential hash.
`code.remote_head` and `code.worktree_dirty` were removed for the same reason — a manifest cannot
meaningfully record the remote state or the tree cleanliness at the moment of its own future commit.

| item | SHA256 |
|---|---|
| original preregistration, `docs/M4_PREREGISTRATION.md` | `7a18ea7258a3a175227a270ee38177c58bde66eadd51af8ccb02c91f83b981af` |
| Amendment 1, `docs/M4_PREREGISTRATION_AMENDMENT_1.md` | recorded in `docs/M4_EXECUTION_MANIFEST.json` |
| **this result document** | recorded in `docs/M4_EXECUTION_MANIFEST.json` |
| result artifact, `runs/m4_formal_result.json` | `f99ac820d9d7853279d0f611d9a68e66fd77b7bfbbbd0f7f068e00f4d0f352af` |

| field | value |
|---|---|
| recorded `preregistration_sha256` **in the artifact** | `7a18ea7258a3a175227a270ee38177c58bde66eadd51af8ccb02c91f83b981af` |
| cohort identity | `e14356b2843835741fd652fbe1b845ebfe951bf968ebba673fb239fe9a325e56` |
| cohort users | 20,000 |
| depth | 100 |
| K values | 5, 10, 20 |
| user message (both arms) | `recommend something useful for me` |
| workers | 4 |

Frozen inputs, hashed in the execution manifest: assembled heads
`da46b8cc31f82fa632173b566f2dfb333c3b5cf7b1704df1b59ef3d6bd580f19`; similar-item index
`e45bbe23ede062046f3740c400d5f2f17c693efeeda49db76ceee97a8aecf551`; cohort targets
`99dbaf7396913303fa1497c908a95e61…`; Two-Tower checkpoint
`0e56be01…`; frozen source modules `loop.py`, `candidate_ledger.py`, `model_policy.py`,
`batched.py`, `split.py` (all unchanged).

Provider: `deepseek` profile, model `deepseek-flash`, base URL `https://api.deepseek.com`,
temperature 0.0, thinking disabled, JSON mode on, `max_tokens` 512. Non-cohort concurrency probe
selected **4-way**; the probe consumed **19 calls against a 16-call budget (3 over)**, a deviation
retained in the manifest rather than reconciled.

## 2. Primary endpoint — NDCG@10

| | value |
|---|---|
| fixed NDCG@10 | **0.01151741511539724** |
| adaptive NDCG@10 | **0.008568399604277856** |
| paired ΔNDCG@10 (adaptive − fixed) | **−0.002949015511119377** |
| recorded 95% CI | **[−0.0041647234398020415, −0.0017272088898192515]** |
| McNemar p | **0.00012875216980012962** |
| paired n | 20,000 users |
| bootstrap | 10,000 resamples, seed `20260201`, unit = user |
| **preregistered verdict** | **SUPPORTED** |

`SUPPORTED` is the preregistered classification label, not a claim about direction. The recorded
verdict rule is: *"SUPPORTED requires every K to have McNemar p<0.05 and both CIs excluding 0;
DIRECTIONAL ONLY when the NDCG delta sign is consistent but not all conditions hold; otherwise NO
DETECTABLE DIFFERENCE."*

**The direction is negative.** Every K satisfies every condition, the confidence interval lies
entirely below zero, and the adaptive arm is worse. This is a **statistically supported negative
result**; read "SUPPORTED" alone it is easy to misread as support for the agent. It is not.

## 3. Secondary endpoints

| K | fixed hits | adaptive hits | gained | lost | ΔRecall | 95% CI | ΔNDCG | 95% CI | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 274 | 200 | 111 | 185 | −0.0037 | [−0.0054, −0.00205] | −0.0028333273562087197 | [−0.004040632966996846, −0.0016389378150144763] | 2.00969874361557e−05 |
| 10 | 401 | 321 | 174 | 254 | −0.004 | [−0.00605, −0.00195] | −0.002949015511119377 | [−0.0041647234398020415, −0.0017272088898192515] | 0.00012875216980012962 |
| 20 | 650 | 509 | 235 | 376 | −0.00705 | [−0.0094, −0.0046] | −0.0037289021944841155 | [−0.004962691925382568, −0.0025361936507599945] | 1.2833371627249196e−08 |

Per-arm metrics as recorded:

| | fixed | adaptive |
|---|---|---|
| Recall@5 / @10 / @20 | 0.0137 / 0.02005 / 0.0325 | 0.01 / 0.01605 / 0.02545 |
| HR@5 / @10 / @20 | identical to Recall (single positive; HR ≡ Recall) | identical to Recall |
| NDCG@5 / @10 / @20 | 0.009465024797456725 / 0.01151741511539724 / 0.014655894807117266 | 0.006631697441248009 / 0.008568399604277856 / 0.010926992612633147 |
| users | 20,000 | 20,000 |
| `mean_target_rank` | 87169.9158 | 91812.25475 |
| `mean_num_candidates` | 156746.0 | 156746.0 |
| catalog size | 156,746 | 156,746 |
| evaluator batches / seconds | 79 / 33.66740973503329 | 79 / 26.24702862103004 |

## 4. Behaviour and source statistics

| field | fixed | adaptive |
|---|---|---|
| users | 20,000 | 20,000 |
| workers | 1 | 4 |
| wall seconds | 1259.628539800644 | 12891.233655691147 |
| users per second | 15.8776967717525 | 1.5514418971973654 |
| peak RSS (GiB) | 1.7488517761230469 | 1.824615478515625 |
| `queried_source_counts` | catalog_search 20000 · history 20000 · similar_item 20000 · two_tower 20000 | **history 19998 · two_tower 16861 · similar_item 170 · catalog_search 0** |
| total tool calls | 80,000 | **37,029** |
| total steps | 100,000 | **57,026** |
| total retries | 0 | 0 |
| failures | 0 | **3** |
| empty rankings | 0 | **2** |
| `protocol_deviation_count` | 0 | **0** |
| `protocol_deviations` | `[]` | `[]` |

Direct arithmetic on recorded fields (no new analysis): mean tool calls per user
**4.0000 → 1.8515**; mean steps per user **5.0000 → 2.8513**; tool-call change **−53.71%**;
NDCG@10 change **−25.60%** relative.

Per the frozen failure rule, the 3 failed and 2 empty-ranking users **remain in the primary
analysis** and are scored on the ranking they actually produced. They were not excluded.

## 5. Interpretation, bounded by what was recorded

**Directly supported.** On the frozen 20,000-user cohort, with the same four-source universe, the
same depth 100, the same ledger RRF, the same shared evaluator and the same 4-call budget, **letting
the model choose which sources to consult and when to stop was significantly worse than a fixed
four-source traversal.** The deficit is consistent and significant at K = 5, 10 and 20.

**Not established causally.** The two arms differ in **two** respects: which sources are consulted,
and **how many**. Fixed issues 4.0000 calls per user; adaptive issued 1.8515, i.e. it used about
46.3% of the retrieval volume, and with depth frozen at 100 that is roughly 185 candidates per user
against roughly 400. A smaller candidate pool mechanically lowers the attainable hit ceiling.
Therefore the measured deficit is the combined effect of **source identity and retrieval budget**,
and this design cannot separate them. Attributing the result to "poor source selection" is not
supported.

**Not established.** Nothing here generalises beyond one model (`deepseek-flash`), one prompt
contract, one temperature (0.0), one cohort and one catalogue. The recorded behaviour — a strong
preference for `history`, frequent `two_tower`, and near-absence of the other two — is a measured
behavioural fact of this policy, not evidence that the policy was wrong to prefer them: M4 ran no
source ablation and no oracle routing, so the incremental value of any single source per user is
unknown.

**Not claimed.** End-to-end live retrieval latency is not claimed. Retrieval was pre-materialised
because the ~3.0 GiB `CatalogSearchSource` and the ~5.9 GiB `SimilarItemIndex` cannot coexist in one
process on the 7 GiB host (preregistration §12.3), so the benchmark's latency figures are
control-plane / trajectory latency and source-call count is the retrieval-cost proxy.

## 6. Known defects and retained deviations

These are recorded, not fixed. Repairing them would mean editing a frozen result artifact, which
would be worse than documenting them.

1. **The artifact records only the original preregistration hash** (`7a18ea72…`). The binding to
   Amendment 1 is carried by `docs/M4_EXECUTION_MANIFEST.json`. A future milestone should record the
   amendment and code hashes *inside* the result artifact.
2. **Per-source counts are not reliable as a strong conclusion.** In this repository the
   `catalog_search` candidate slot is served by `FrozenQueryCatalogSearch`, a dual-identity object
   that also carries similar-item semantics, so adaptive's 170 registrations under `similar_item`
   are not interpretable as similar-item behaviour. Read `catalog_search: 0` as "this source was
   effectively not selected", not as a literal count of catalogue searches.
3. **The concurrency probe exceeded its budget:** 19 calls against 16, because a confirmation run
   was counted after the fact. Recorded in the manifest and in commit `0e6a7e8`.
4. **The pre-amendment 12-user adaptive preflight is invalid as an efficacy result.** It measured
   the broken action contract (11 of 12 users failed, 10 empty rankings) and must never be quoted as
   a quality finding. It is retained only as before-fix evidence in Amendment 1 §4.
5. **Manifest timing circularity — recorded, and its root cause repaired at seal time.** The
   manifest was regenerated while the formal run was in progress, so its per-file code hashes
   describe the working tree during the run rather than a snapshot frozen before it. The code did
   not change during the run, so this holds semantically, but the strict ordering should be: freeze
   protocol → freeze amendment → freeze code hashes → execute.

   Two structural causes were found while sealing and are now fixed rather than merely noted, since
   both were defects in the *manifest tooling*, not in the experiment:

   - The manifest derived `code.commit` from `git rev-parse HEAD`. Writing the manifest changes the
     working tree, so committing it produced a commit the field could never name; the manifest was
     therefore stale the instant it was committed, and each regeneration rewrote the field to the
     previous commit. The field is now the pinned constant `code.executed_code_commit` — renamed
     from the ambiguous `code.commit` — naming the commit that carried the code the run executed.
     The seal commit is expressed as the Git ref `code.seal_ref` (`M4-sealed`) rather than an
     embedded hash, because embedding it is exactly the self-reference that caused the defect.
     `code.remote_head` and `code.worktree_dirty` were removed for the same reason: a manifest
     cannot record the remote state or the tree cleanliness at the moment of its own future commit.
   - `--check` rewrote both manifest files, so every invocation dirtied the tree and the check could
     never be idempotent — a verification that edits the thing it verifies is not a verification.
     `--check` now writes nothing. Verified by running it three times: zero working-tree changes
     each time.
6. **Protocol defects found during preflight and repaired before execution** (see Amendment 1 §3 and
   §9): a blank `SELECT_SOURCE{catalog_search}` query silently returned zero candidates; the
   projected schema omitted `select_source`'s legal values, so the model answered `"default"` in
   every probe; and a duplicate-source deviation escaped `choose()` as an execution failure instead
   of terminating on the held ranking. All three were fixed **before** the formal cohort ran.

## 7. What was verified

- Cohort: 20,000 users, identity `e14356b2…`, exact frozen order, verified against the frozen M3
  `target_ids`; all four source heads complete at 20,000 users × depth 100.
- Execution: both arms ran all 20,000 users; 0 protocol deviations; 0 retries; 3 failures and 2
  empty rankings in the adaptive arm, all retained in the analysis.
- Evaluation: one shared evaluator call per arm after both arms completed, over the full 156,746-item
  catalogue, with the frozen protocol, seen-item masking and deterministic tie-breaking.
- Tests at seal time: the M4 and control-plane suites pass (259 passed, 0 failed).
- Manifest check is a pure verification: `--check` writes nothing and repeated runs leave the working
  tree unchanged (verified by three consecutive runs with zero modifications).
- **Credential history audit: CLEAN.** See §8.

## 8. Credential history audit

Run with `python -m experiments.credential_audit`, which is committed so the check is repeatable
rather than a one-off. It is deliberately stronger than a filename search.

**Method.** Every blob in the object database is scanned via `git cat-file --batch-all-objects`, not
just reachable history — a credential that was committed and later removed still sits in the object
database until garbage collection prunes it, and a history walk would miss it. Two passes run over
those blobs:

1. a literal search for the **real** value from the gitignored `.m4_credential`;
2. a search for common credential **shapes**: provider keys (`sk-…`, `sk-ant-…`), bearer tokens, AWS
   access key ids, GitHub / Google / Slack tokens, private-key headers, JWTs, and generic
   `secret = …` assignments.

**Result: no leak.**

| check | result |
|---|---|
| blobs scanned (incl. unreachable objects) | 737 |
| **real credential value hits** | **0** |
| secret-shape hits | 16 |
| of those, inside expected synthetic-fixture context (`tests/`) | 16 |
| shape hits outside fixture context | 0 |
| `.env.example` contains the real value | no |
| verdict | **CLEAN** |

The 16 shape hits are all in `tests/test_memory_service.py`, `tests/test_packaging.py` and
`tests/test_provider_record_replay.py`, and they are **synthetic fixtures by construction**:
`test_secret_like_shapes_are_rejected` is parametrised over one example per shape the credential
detector must catch (its comment reads "each is a synthetic fixture"), and
`test_provider_record_replay.py` asserts that a fake key does **not** appear in a logged settings
description. The fake values were confirmed different from the real credential, and the earliest of
them entered at `c03582b` ("milestone9: add preference memory"). No real value appears anywhere.
Per the redaction rule, only a few leading characters of any match are ever reported.

`runs/m4_formal_result.json` and `.m4_credential` are both gitignored, so neither can enter history
by accident; the audit confirms neither ever did.
