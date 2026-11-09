# CoShop external benchmark adapter (M6.2, optional / offline lab only)

CoShop is an **external** conversational-shopping benchmark.  AgentRec-X is the subject
being evaluated; CoShop is the measuring instrument.  This directory is an optional
adapter that lets the AgentRec-X **offline lab** reuse the official CoShop task set, the
official visible-query retrieval baseline and the official CoShop metric implementation.

It is not part of the recommendation system.  It never runs on the serving path, never
changes Serving V1 / RRF / the accepted reranker / M6.1 ranking code, and no production or
core-lab module imports it.

---

## 1. Isolation boundary

```text
experiments/external_benchmarks/coshop/   (this directory; optional, deletable)
        |  lazy import from an EXTERNAL checkout (never vendored)
        v
C:\...\coshop-main\  (official CoShop repository, own licence/environment/data)
        |
        |  duck-typed reads:
        |    spec.z0 / spec.initial_known_features   -> public visible query
        |    spec.z0_baseline_ids                     -> official baseline candidates
        |    spec.xstar                               -> SEALED HiddenTaskTargets (evaluator only)
        v
AgentRec-X offline contract (imported lazily inside the conversion):
        recommendation.ranking.schemas.CandidateSnapshot / CandidateRecord
```

Hard rules:

1. **No vendoring.**  The upstream CoShop repository carries no licence, so neither its
   source nor its assets are copied into AgentRec-X.  The adapter reads an external
   checkout; removing that checkout leaves AgentRec-X untouched.
2. **No import-time coupling.**  `import experiments.external_benchmarks.coshop` imports
   only the Python standard library: no CoShop, pandas, torch or AgentRec-X modules.
   CoShop is imported lazily inside an actual evaluation; the M6.1 snapshot contract is
   imported lazily inside the snapshot conversion.
3. **No hidden-target leakage.**  A public task view holds only the user-visible query
   (`z0` / initial known features) and the official visible-query candidate ids.  The
   hidden target (`xstar`) is extracted into a separate sealed `HiddenTaskTargets`
   object that only the evaluator/diagnostics touch.  It is never passed to the snapshot
   conversion, the ranking callable or the report.  The public task view, snapshot,
   ranked ids and report payload are serialisation-tested against a target sentinel.
4. **No target-synthesised candidates.**  Candidates are exactly the official
   `spec.z0_baseline_ids` (deduplicated, order preserved).  A missing/empty baseline is a
   typed `CoshopContractError`; candidates are never rebuilt from targets.
5. **No identity mixing.**  Candidate identities are official CoShop external id
   strings.  The AgentRec-X `item_id` is a deterministic 1-based **catalog-order
   position** local to this external benchmark; no Amazon `parent_asin`/`item_id` value
   can appear in these snapshots (`metadata_status="missing"`, no metadata row).
6. **No fabricated metrics.**  Official metrics are copied from CoShop's own
   `compute_evaluation_metrics`; custom retrieval diagnostics are labelled as custom;
   dialogue/decision metrics are explicit `NOT_EXECUTED`.  If the external checkout, its
   dependencies or the official Google Drive data assets are unavailable, the adapter
   fails with `CoshopUnavailable` and the CLI exits `2` printing
   `NOT EXECUTED: <reason>` — it never reports zeros or guessed numbers.

---

## 2. External setup (not performed by this repository)

The adapter does **not** install CoShop and does **not** download data itself.  To run a
real official evaluation, prepare an external checkout:

1. Obtain the official CoShop repository and check it out **outside** the AgentRec-X tree
   (e.g. `C:\...\coshop-main`).  Do not copy it into this repository (no licence).
2. Create a **separate** virtual environment for CoShop and install its own
   `coshop/requirements.txt` there.  Do not add CoShop dependencies to AgentRec-X's
   requirements.
3. Download the official Google Drive data assets **once**, following the CoShop README,
   into `coshop/data/<domain>/assets/` inside that checkout (the official
   `Dataset._ensure_assets_available` handles the download when the Drive files are
   reachable).  This environment currently cannot reach those assets, so no real run has
   been executed here (see §7).
4. Point the adapter at the checkout with `--coshop-root <path>` or the environment
   variable `AGENTRECX_COSHOP_ROOT=<path>`.

The adapter validates that the configured root contains `coshop/data/dataset.py`; a wrong
path fails fast with an actionable `CoshopUnavailable`.

---

## 3. Commands

Run from the AgentRec-X repository root, using the AgentRec-X test virtual environment
(the adapter itself is standard-library-only; the AgentRec-X offline contract import is
torch-free):

```bash
# NOT EXECUTED is expected unless the external checkout + official assets exist
python -m experiments.external_benchmarks.coshop.run_evaluation \
    --domain movielens --version v2 --split test --tasks 3 --k 5 \
    --coshop-root /path/to/coshop-main --json runs/m6_2_coshop_report.json

# without --coshop-root the checkout comes from AGENTRECX_COSHOP_ROOT
python -m experiments.external_benchmarks.coshop.run_evaluation --tasks 3 --k 5

# focused isolation/contract tests (no network, no CoShop installed)
python -m pytest -q tests/test_m6_2_coshop_adapter.py
```

Exit codes: `0` executed (report with digest written when `--json` is given), `2`
`NOT EXECUTED: <reason>` (checkout/dependency/assets unavailable or contract violation;
the JSON file, if requested, contains the `NOT_EXECUTED` status and no metrics), `1`
unexpected internal error.

Programmatic use:

```python
from experiments.external_benchmarks.coshop import evaluate_coshop

report = evaluate_coshop(domain="movielens", version="v2", split="test",
                         limit=1, k=5, coshop_root="/path/to/coshop-main")
print(report.report_digest, report.official_metrics["recall_at_k"])
```

`dataset_factory=` / `metrics_function=` are injection seams for tests and for other
official evaluators; with both injected no CoShop checkout is needed at all.

---

## 4. Metric table (official vs custom vs NOT_EXECUTED)

| Metric | Source | Status |
| --- | --- | --- |
| `recall_at_k` / `recall_at_1` / `ndcg_ustar_at_k` | official `coshop.evaluation.metrics.compute_evaluation_metrics` (`compute_embedding_ustar=False`) | **EXECUTED** for the official visible-query baseline |
| Candidate-set size, `retrieval_calls`, `dialogue_turns`, `clarifying_questions`, `tool_calls` | adapter | **CUSTOM** diagnostics (`custom_retrieval_baseline`) |
| `target_in_candidate_pool`, `target_original_rank_in_pool`, `custom_recall_at_k` | adapter (sealed-target diagnostics) | **CUSTOM** diagnostics, **not** benchmark metrics |
| `team_accuracy` (official user-simulator rerank `recall_at_1`) | CoPref user simulator + hosted LLM API | **NOT_EXECUTED** — never replaced by top-1/retrieval `recall@1`; no simulated-user result fabricated |
| `conditional_selection_accuracy` | CoPref user simulator + hosted LLM API | **NOT_EXECUTED** |
| `interaction_cost` (turns/questions/tokens/runtime) | multi-turn official episode | **NOT_EXECUTED** — this adapter runs one retrieval call, zero dialogue turns |

The executed thing is a **retrieval baseline**: the official visible-query baseline
candidate order.  It is explicitly **not** an AgentRec-X model and **not** an M6.1
A0/A1/A2 ranking arm.  Ranking change or recall numbers from this adapter are never
reported as AgentRec-X quality evidence.

---

## 5. What the AgentRec-X snapshot looks like

One snapshot per official task, using `recommendation.ranking.schemas` only:

* `candidate_id` = the official external id string (verbatim);
* `item_id` = 1-based position in the official catalog order (deterministic, distinct);
* `original_rank` = 1..n in official baseline order;
* `score = 0.0`, `score_kind = "coshop_official_baseline_order"` (provenance only);
* `metadata_status = "missing"`, `metadata = None` (no Amazon catalogue row is invented);
* `eligibility = not_assessed` (the external benchmark carries no AgentRec-X constraint
  assessment);
* `source = "coshop_external_benchmark"`;
* snapshot/task/user/turn ids are derived from `coshop:<domain>:<version>:spec-NNNNNN`.

The ranking callable is `rank_baseline_candidates(task_view, candidate_snapshot)`: it
receives only those two objects and only reorders the frozen candidate set (by default it
preserves the official order).  Any candidate-set change is a contract error.

---

## 6. Reproducibility

Every report carries deterministic provenance and a `report_digest` (SHA-256 over the
canonical report body): domain/version/split, requested task limit and evaluated task
indices, `k`, baseline name, seed (`0`; no sampling is performed, recorded for protocol
reproducibility), catalog id count and SHA-256 of the ordered catalog ids, adapter
version, protocol label `agentrecx.m6_2.coshop_external_baseline.v1`, and the CoShop
package version/root when a real checkout is used.  Two calls with the same inputs
produce byte-identical reports and digests; results are reproducible only when the same
external checkout, official asset version and catalog are used.

## 7. Current environment limitation (recorded, not a result)

This worktree was developed offline.  The official CoShop data assets are hosted on
Google Drive and are **unreachable from this environment** (confirmed timeouts) and the
external checkout present on this machine has no downloaded assets.  A real official run
was therefore **not possible** here: the adapter and CLI fail cleanly with
`CoshopUnavailable` / `NOT EXECUTED` instead of producing numbers.  The unit tests use
duck-typed fake dataset/spec/catalog objects and an injected fake official-metrics
function, so they need no network and no CoShop installation.  The optional real-data test
is skipped unless `AGENTRECX_COSHOP_ROOT` is set.

## 8. Deletion safety

Delete `experiments/external_benchmarks/` (and its test file) and AgentRec-X is
unchanged: no production module, no core-lab module (`recommendation/`), no M6.1 file and
no existing test imports anything from this directory.  An automated test scans
`recommendation/`, `agent/` and `api/` for reverse dependencies.
