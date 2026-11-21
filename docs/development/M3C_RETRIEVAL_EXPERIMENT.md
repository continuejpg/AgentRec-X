# AgentRec-X M3-C — BM25 Retrieval Experiment

> Status: one completed baseline/candidate experiment. No production retrieval change.
> The candidate is promising on a 10-query fixture, but its labels are machine-assisted and
> pending human review, so integration is deliberately deferred.

## 1. Question

M3-B showed that the current full-catalogue BM25 top-5 is dominated by accessories for laptop
requests (chargers, adapters, batteries, stands, sleeves, keyboard covers). This experiment
tests one candidate strategy against the **current production recommendation retrieval
baseline**:

- **Baseline:** the effective query that `recommend_products` already constructs with
  `augment_query(raw_query, extract_task_requirements(raw_query))`, searched over the full
  catalogue BM25 index.
- **Candidate:** the **same effective query** and the **same BM25 parameters**, searched over a
  catalogue subset scoped to the requested product category. The scope is inferred from the
  query and existing title/sidecar category metadata.

The one experimental variable is **retrieval scope**. Query text, BM25 k1/b, top_k, catalogue,
sidecar, and price mode are held constant.

This also records an important distinction from M3-B: M3-B measured the BFF display
`search_products` raw query; M3-C measures the production recommendation query that already
contains the structured category. The baseline remains the actual current recommendation
retrieval path, not a deliberately weakened one.

## 2. Query set and labels

Ten queries cover the requested categories:

| Query | Target | Use case/feature |
|---|---|---|
| `laptop_5000` | laptop | — |
| `laptop_gaming_8000` | laptop | gaming |
| `laptop_office_6000` | laptop | office |
| `laptop_gaming_1000` | laptop | gaming |
| `headphones_noise` | headphones | — |
| `headphones_gaming_500` | headphones | gaming |
| `headphones_noise_500` | headphones | noise cancelling |
| `tablet_study_3000` | tablet | study |
| `mouse_wireless_300` | mouse | wireless |
| `keyboard_mechanical_500` | keyboard | mechanical |

For each query the runner retrieves top-5 under both strategies, takes the **union** of the
two top-5 lists, and labels every pooled id. The label rule uses product title, target
category, accessory terms and explicit use-case/feature terms. It never reads a BM25 score or
rank.

**Labels are machine-assisted by the Harness and pending human review; they are not an
independent human annotation.** The output includes every judged title and reason so a human
can review or replace the labels:

- `runs/retrieval_experiments/<run_id>/annotations.json`
- `runs/retrieval_experiments/<run_id>/annotations.md`

Only binary labels are used. Recall/MRR/nDCG are intentionally not reported: the judged pool
contains only the top-5 union, not all relevant products in the 76,263-record catalogue.

## 3. Recorded result (`m3c-pre-09`)

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Precision@5 | **11 / 50 = 22.0%** | **37 / 50 = 74.0%** | **+52.0 pp** |
| Obvious error/accessory count | 39 / 50 | 13 / 50 | -26 |
| Error/accessory share | 78.0% | 26.0% | -52.0 pp |
| Search-only latency, sum of per-query medians | 435.97 ms | 6.84 ms | — |

Per-query:

| Query | Baseline relevant@5 | Candidate relevant@5 | Status |
|---|---:|---:|---|
| `laptop_5000` | 0 | 5 | improved |
| `laptop_gaming_8000` | 0 | 5 | improved |
| `laptop_office_6000` | 0 | 1 | improved |
| `laptop_gaming_1000` | 0 | 5 | improved |
| `headphones_noise` | 1 | 4 | improved |
| `headphones_gaming_500` | 4 | 5 | improved |
| `headphones_noise_500` | 4 | 5 | improved |
| `tablet_study_3000` | 0 | 1 | improved |
| `mouse_wireless_300` | 2 | 4 | improved |
| `keyboard_mechanical_500` | 0 | 2 | improved |

Candidate scope sizes: laptop 457, headphones 1,993, tablet 339, mouse 377, keyboard 559.
Scope index build was 12–65 ms per category; sidecar loading was about 8.0 s and is shared with
the existing catalogue setup. These are separate from the per-query search latency above.

The laptop gaming top-5 changed from keyboard skins/covers and a headset to real gaming
laptops (Alienware, MSI, ASUS, HP Omen, Lenovo Y50). The general laptop top-5 changed from
laptop stands to real ASUS/Lenovo/MSI laptops. The office and tablet queries improved in
category purity but only reached 1/5 because their remaining candidates do not satisfy the
explicit use-case label (office/study).

## 4. Interpretation

The candidate has a clear, explainable effect **on this fixture**: it removes the dominant
accessory class and lets BM25 rank within the target category. It also reduces search time
because the BM25 corpus is smaller.

It is **not integrated** in this milestone because:

1. the labels are machine-assisted and have not been independently human-verified;
2. the scope classifier is still heuristic and has false positives/negatives (for example a
   laptop portfolio or a tablet cradle can survive a broad title rule);
3. the test is 10 queries and 50 judged slots per strategy, not a representative benchmark;
4. no broad-query regression or downstream recommendation evaluation was run for an
   integrated candidate.

These limits are deliberate. The experiment does not establish statistical significance, and
does not claim that the candidate will improve queries outside the selected categories.

## 5. Downstream behavior

No production retrieval, Budget Gate, requirement merge, Finalize or Replay code was changed.
The current M3-B downstream result therefore remains:

- Task Outcome Accuracy **13 / 15**;
- Budget violations **0 / 12**;
- Correct refusals **2 / 2**.

If a future milestone integrates a category scope, it must first obtain human-reviewed qrels
and run the M3-B task suite plus M1/M2/M3-A regressions on the integrated path.

## 6. Reproduce

```bash
.venv/bin/python -m agent_eval.retrieval \
  --queries agent_eval/retrieval/queries_v1.json \
  --output-dir runs/retrieval_experiments
```

The experiment records Git SHA, query-set SHA256, catalogue/sidecar SHA256, BM25 k1/b,
top_k, repeats, scope sizes, per-query top-5 and metadata in `report.json`/`report.md`.

One factual metadata correction accompanies this commit: M3-B's report writer had hard-coded
`bm25_k1=1.5`; the actual accepted constant is `1.2` (`recommendation.control.catalog_search`).
The M3-C experiment reads the constant directly. No retrieval behavior changed.
