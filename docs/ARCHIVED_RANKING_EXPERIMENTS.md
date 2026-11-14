# Archived Ranking Experiments

> Status: **ARCHIVED / NOT MAINTAINED**
> Decision: stop optimizing the LightGBM rating ranker and Neural Memory Ranker. Keep only this concise record and the reusable data/eval assets. Production default remains `PreferenceAwareScorer`; no shadow model is wired into production.

---

## 1. LightGBM OFF / ON

### Data and task

- Source: Amazon Reviews 2023 Electronics rating-only, catalog-filtered.
- Dataset: `data/memory_ranking_v1/` (Phase 2).
- Phase 2 targets: train 22,427 / val 14,157 / test 14,157.
- Users: 14,157 with ≥3 catalog interactions; items used: 7,147.
- History: catalog-only; target excluded; temporal split.
- Features:
  - Memory OFF: 95 item/popularity features.
  - Memory ON: 213 features = 95 + 118 `inferred_from_ratings` memory features.
- Training: LightGBM 4.7.0 CPU, 600 trees, LR 0.05, leaves 31, early stopping.
- Models: pointwise rating regression + high/low classification; no unobserved negatives.

### Key metrics (test)

| Task | Memory OFF | Memory ON |
|---|---:|---:|
| Rating MAE | 0.855586 | 0.756594 |
| Rating RMSE | 1.135553 | 1.065706 |
| High/Low ROC-AUC | 0.571768 | 0.749143 |
| High/Low PR-AUC | 0.917892 | 0.957258 |
| Observed Pairwise (Phase 2 protocol) | 0.569103 | 0.586711 |
| High-vs-Low Pairwise | 0.594005 | 0.613079 |

Phase 2 report: `docs/MEMORY_RANKING_V1_REPORT.md`; dataset manifest `data/memory_ranking_v1/manifest.json`.

### Later expanded LightGBM reference

An expanded-data pointwise rating baseline (90 / 228 features) was also tested in Phase 3 on the new sampled candidate protocol:

```text
category-matched common 14,157:
  Memory OFF: Hit@10 0.521368, nDCG@10 0.266235, MRR 0.199477, pair 0.592930
  Memory ON:  Hit@10 0.594335, nDCG@10 0.295687, MRR 0.216107, pair 0.592740
```

This was not a production candidate; it was a comparison baseline.

---

## 2. Neural Memory OFF / ON

- Dataset: `data/neural_memory_v1/` (Phase 3).
- Training rows: 43,435; val 14,157; test 77,029.
- Full Electronics history, catalog-only targets; 8,029,455 raw interactions.
- Model:
  - 90 item/popularity features, 92 memory features, hidden 128.
  - item encoder + masked history attention + candidate-conditioned attention.
  - attribute match/conflict interaction (23 attributes).
  - no user_id / raw item-id embedding.
  - Memory OFF 163,203 params; Memory ON 325,251 params.
- GPU: RTX 4090 D, 24GB; PyTorch 2.11+cu128.
- Training: 8 epochs, 32 sampled negatives, batch 256.
- Cost: OFF 6.867s / 201.8MB peak; ON 15.043s / 458.8MB peak.
- Negative sampling: 1 real observed positive + 99 sampled unobserved catalog candidates; not confirmed negatives.

### Primary category-matched sampled metrics

| Metric | Memory OFF | Memory ON |
|---|---:|---:|
| Full test Hit@10 | 0.916278 | 0.918407 |
| Full test nDCG@10 | 0.770639 | 0.775925 |
| Full test MRR@10 | 0.727636 | 0.733822 |
| Common 14,157 Hit@10 | 0.905347 | 0.910221 |
| Common 14,157 nDCG@10 | 0.737407 | 0.743299 |
| Common 14,157 MRR@10 | 0.688000 | 0.694111 |
| Observed Pairwise | 0.509692 | 0.515964 |
| High-vs-Low Pairwise | 0.511196 | 0.521433 |

Attribute ablation: removing candidate-attribute interaction reduced nDCG 0.775925→0.773006, MRR 0.733822→0.730565, pairwise 0.515964→0.507412, high-vs-low 0.521433→0.504798.

Phase 3 report: `docs/MEMORY_CONDITIONED_GPU_V1_REPORT.md`; data manifest `data/neural_memory_v1/manifest.json`.

---

## 3. Observed Pairwise Results

| Model / protocol | Pairwise ordering | High-vs-Low |
|---|---:|---:|
| Phase 2 LightGBM OFF | 0.569103 | 0.594005 |
| Phase 2 LightGBM ON | 0.586711 | 0.613079 |
| Phase 3 expanded LightGBM OFF | 0.592930 | 0.625080 |
| Phase 3 expanded LightGBM ON | 0.592740 | 0.622521 |
| Neural OFF | 0.509692 | 0.511196 |
| Neural ON | 0.515964 | 0.521433 |

Neural pairwise was near random; pointwise next-item gains did not translate into stable user-specific pairwise ordering.

---

## 4. P20 Shadow

P20 fixed Original-BGE Top-20; provisional acceptable labels; no training.

| Arm | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|
| Relevance only (Phase 1 A) | 14/19 | 19/19 | 19/19 |
| Existing Scorer Memory OFF (Phase 1 B) | 14/19 | 19/19 | 19/19 |
| Existing Scorer Memory ON (Phase 1 C) | **18/19** | 19/19 | 19/19 |
| Neural OFF | 15/19 | 18/19 | 19/19 |
| Neural ON | 13/19 | 17/19 | 19/19 |
| Neural ON + BGE blend | 14/19 | 18/19 | 19/19 |

Neural shadow was **below** the existing Scorer Memory ON; top-1 changed 15/19; no Gate replacement was attempted.

---

## 5. Why the Old Models Did Not Migrate

1. **No query-relevance training feature.** LightGBM and neural rankers used item/rating history, not the Agent's BGE query signal.
2. **Event-Sourced Memory ≠ inferred rating history.** P20 shadow required a synthetic pseudo-history; this was OOD for the neural model.
3. **P20 shadow below existing Scorer.** 13/19 vs 18/19; no production case.
4. **Pairwise near random.** Next-item sampled ranking gains did not create reliable user preference ordering.
5. **Popularity dominated sampled candidates.** In Phase 3, popularity Hit@10 was 0.95–0.97, above neural models.
6. **Metadata coverage too low.** Only 27,403 catalog interaction items had metadata; 70% of expanded history events were non-catalog UNKNOWN items.
7. **Data labels are not human pairwise preference.** Ratings/observed interactions cannot be promoted to explicit memory gold.
8. **P20 labels provisional.** A 19/19 result would not prove production superiority.

Decision: **archive these models; do not continue LightGBM/Neural Memory Ranker optimization.**

---

## 6. Git Commits and Data Sources

| Commit | Content |
|---|---|
| `4ac9f67` | Phase 2 dataset + CPU LightGBM baselines |
| `76fdffb` | per-target history metadata addition |
| `1ab16d9` | Phase 3 GPU neural ranker + shadow |
| `30129d8` | Phase 3 report + ItemCF/SASRec non-run note |
| `5323a41` | Phase 1 Memory Contribution benchmark |

Data sources:

- `data/gold/amazon_reviews_electronics_v1/electronics_catalog_ratings.csv`
- `data/memory_ranking_v1/` (Phase 2 dataset, kept as reusable data)
- `data/neural_memory_v1/` (Phase 3 expanded sequence data, kept as reusable data)
- `data/enrichment/electronics_metadata_sidecar.jsonl`
- `data/p13r/electronics/catalog_meta.jsonl`
- P20: `runs/p20_agent_eval/task_suite.json`
- Phase 1 results: `runs/memory_contribution_phase1/`

---

## 7. Cleanup Record

### Deleted local files

Directory `runs/memory_ranking_v1/` (obsolete LightGBM / pointwise outputs):

```text
runs/memory_ranking_v1/comparison.json
runs/memory_ranking_v1/feature_importance.json
runs/memory_ranking_v1/memory_off_classifier.txt
runs/memory_ranking_v1/memory_off_model.txt
runs/memory_ranking_v1/memory_on_classifier.txt
runs/memory_ranking_v1/memory_on_model.txt
runs/memory_ranking_v1/ranking_group_metrics.json
runs/memory_ranking_v1/test_predictions.parquet
runs/memory_ranking_v1/training_summary.json
```

Total: **1,722,157 bytes (9 files)**.

Neural `.pt` weights and LightGBM expanded weights in `runs/neural_memory_v1/`:

```text
runs/neural_memory_v1/memory_off.pt
runs/neural_memory_v1/memory_on.pt
runs/neural_memory_v1/memory_off_hard.pt
runs/neural_memory_v1/memory_on_hard.pt
runs/neural_memory_v1/memory_off_hard_noattr.pt
runs/neural_memory_v1/memory_on_hard_noattr.pt
runs/neural_memory_v1/expanded_lightgbm_off.txt
runs/neural_memory_v1/expanded_lightgbm_on.txt
```

Total: **5,931,509 bytes (8 files)**.

Model training logs:

```text
runs/neural_memory_v1/remote_logs/
```

Total: **54,934 bytes (7 files)**.

Local released: **7,708,600 bytes (~7.35 MiB)**.

### Deleted remote files (AutoDL)

Same 8 model weight files under:

```text
/root/autodl-tmp/AgentRec-X-phase2/runs/neural_memory_v1/
```

Released: **5,931,509 bytes (~5.66 MiB)**.

Remote data `data/neural_memory_v1` was **retained** as reusable real sequence/evaluation data. No raw data, metadata, BGE, Qwen, P14/P18/P20, Event-Sourced Memory, Scorer/Gate, or future asset was deleted.

### Retained archive evidence

Small compact JSON summaries/remote logs were retained where useful for audit:

- `runs/neural_memory_v1/comparison*.json`
- `runs/neural_memory_v1/lightgbm_comparison*.json`
- `runs/neural_memory_v1/neural_subsets.json`
- `runs/neural_memory_v1/shadow_p20.json`
- `runs/shopranker_v1/shopranker_results.json` (new direction)
- `data/memory_ranking_v1/` and `data/neural_memory_v1/` (reproducible data assets)

### Not deleted

- Amazon Reviews / ESCI raw or gold data
- Electronics Catalog / Metadata Sidecar
- Original BGE index/vectors
- Event-Sourced Memory stores
- P14 / P18 / P20
- production Scorer / Gate / Planner
- Qwen3.5 Base / Qwen3-Reranker-4B / BGE model
- ShopRanker model (new active experiment)
- future feature/eval assets
