# Preference-Aware Candidate Ranking v1 — Phase A Report + GPU Handoff

## Status

- **Phase A local CPU benchmark: complete.**
- **Phase B frozen Qwen3-Reranker-4B: not executed.** The AutoDL endpoint
  `connect.cqa1.seetacloud.com:36856` was offline (`Connection refused`) at the time of
  this run. No Qwen3 ranking numbers are claimed here.
- User selected the fallback: deliver local ranking results and a reproducible GPU handoff.
- No Planner fine-tuning, retrieval change, Gate change, or ranking training was performed.

## 1. Final selection chain and actual control boundary

The current path is:

```text
Original BGE search (Top-20)
  -> PreferenceAwareScorer candidate_rows
  -> Executor role/task-constraint filtering
  -> DecisionGate consumes presentable rows
  -> Gate selects presentable[0]
```

Details verified in code:

- BGE supplies the query rank/score.
- `PreferenceAwareScorer` uses:
  - 0.50 query relevance (rank-based within fixed pool);
  - 0.20 per matched positive preference, capped at 3;
  - 0.35 per conflicting negative preference;
  - a 0.10 unknown-hard-constraint penalty;
  - hard tiers: verified (0) > unknown (1) > violated (2).
- Hard violations are forced below non-violators and marked unpresentable.
- The executor filters role-compatible rows and current-task hard exclusions into
  `last_rows`.
- RERANK calls the same scorer/inspection and sets the candidate order to the hard-filtered
  eligible rows; it does **not** select a product itself.
- The Gate is not merely a validator: it filters hard/role-incompatible rows and selects the
  first presentable candidate. In this fixed-pool benchmark, **Gate selection equals the
  hard-filtered scorer top-1 for 19/19 tasks**.

Thus the order the Planner sees after RERANK and the Gate's chosen product are both driven by
the same structural filters + scorer; Qwen's Planner does not output item IDs.

## 2. Fixed candidate dataset

Artifacts:

```text
runs/preference_ranking_v1/ranking_dataset.jsonl
runs/preference_ranking_v1/manifest.json
runs/preference_ranking_v1/local_summary.json
```

Construction:

- 30 P20 tasks; 19 recommend tasks with evaluator acceptable sets;
- fixed pool: **20 candidates** from the first real Original-BGE retrieval in each P20
  trace, in the original BGE order;
- query: task natural-language query;
- active preferences: legal Memory-reader output only;
- historical memory ids/turns retained;
- hard requirements: `prepare_hard(current_query)` only;
- product evidence: title, catalogue text, attributes, form factor;
- eligibility/scorer row: hard tier, role status, matched/conflicting/unknown preferences,
  presentable flag;
- acceptable ids: joined from evaluator **after** ranking, for measurement only;
- all other candidates are `unjudged`, never treated as negatives.

Data SHA256:

```text
166ab7f523644e14355cbd7ad10e4053d389da2fa99c78bcbc1149cc29beadb6
```

Label status:

```text
provisional_automated_verified_not_human_gold
```

No P18 Reserved task was used. Eval100 tasks are included in this no-training dataset for
diagnosis, but no fitting or hyperparameter selection was performed.

## 3. Local ranking metrics (19 recommend tasks)

| Variant | Hit@1 | Hit@5 | Hit@10 | MRR | Hard violations | Preference-matched top1 | Preference-violated top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A Original BGE order | 12/19 | 16/19 | 17/19 | 0.7379 | 0 | 12 | 2 |
| B Existing scorer, unfiltered | 15/19 | 16/19 | 18/19 | 0.8181 | 0 | 14 | 0 |
| **C Structural hard filter + scorer** | **18/19** | **19/19** | **19/19** | **0.9737** | **0** | 13 | 0 |

Additional facts:

- Candidate coverage: **19/19** recommend tasks have at least one acceptable candidate in
  the fixed pool.
- All 19 fixed pools contain 20 candidates.
- UNKNOWN soft preference at top-1: 10/19 for every variant; the Gate currently permits
  finalizing with soft unknown evidence.
- Active-preference subgroup (14 tasks): A 8/14, B 11/14, C 13/14 Hit@1.
- No-active-preference subgroup (5 tasks): A 4/5, B 4/5, C 5/5 Hit@1.
- Hard-requirement subgroup (7 tasks): A 6/7, B 6/7, C 7/7 Hit@1.
- Gate selected a product in the fixed pool for 19/19 tasks and selected C top-1 for 19/19.
- Gate selected an acceptable product for 18/19; only `0051` fails.

Interpretation:

1. **A -> B** shows the existing scorer improves over raw BGE order.
2. **B -> C** is not a new scoring idea; it is the structural hard/role filter already
   present in the executor. C is the effective baseline the Gate consumes.
3. Therefore **there is almost no current end-to-end ranking headroom in these 19 tasks**:
   C/Gate already has 18/19 acceptable top-1 and full Hit@5/Hit@10.

## 4. The one remaining local failure: 0051

```text
Query: "Which earbuds work for workouts? I'd rather hear one clear suggestion than a long list."
Active memory:
  negative over_ear; positive rechargeable; positive microphone
Hard requirements: none
Acceptable id in pool: B08D32JM5F
BGE rank of B08D32JM5F: 16
C/Gate selected: B07BKPKXTT
B07BKPKXTT label: unjudged (not a claimed negative)
B07BKPKXTT: in-ear, wireless, no matched preferences; rechargeable/microphone unknown
B08D32JM5F: in-ear, wireless, microphone present; rechargeable unknown
C first acceptable rank: 2
A top1: B0777TCMGN
B top1: B07D8RXD5R
C top1: B07BKPKXTT
```

This is the only local case where a candidate already in the pool is acceptable but the
current scorer/Gate top-1 is not. It is a plausible frozen-reranker target, but it is one
case, not a ranking-training set.

## 5. Frozen Qwen3-Reranker-4B handoff (Phase B, not executed)

Reuse existing model:

```text
/root/autodl-tmp/models/Qwen3-Reranker-4B
```

Input already built locally; upload it and the GPU script:

```bash
# from the developer workspace
tar -czf /tmp/preference_ranking_v1.tar.gz \
  runs/preference_ranking_v1/ranking_dataset.jsonl \
  runs/preference_ranking_v1/manifest.json \
  experiments/preference_ranking/run_qwen3_reranker.py \
  experiments/preference_ranking/combine_qwen3.py

scp -P <PORT> -i <AUTODL_KEY> /tmp/preference_ranking_v1.tar.gz \
  root@<AUTODL_HOST>:/tmp/
```

On GPU:

```bash
cd /root/autodl-tmp/AgentRec-X-phase2
mkdir -p runs/preference_ranking_v1 experiments/preference_ranking
tar -xzf /tmp/preference_ranking_v1.tar.gz
PYTHONPATH=. /root/autodl-tmp/bge_env/bin/python \
  experiments/preference_ranking/run_qwen3_reranker.py \
  --data runs/preference_ranking_v1/ranking_dataset.jsonl \
  --out runs/preference_ranking_v1/qwen3_scores.jsonl \
  --model /root/autodl-tmp/models/Qwen3-Reranker-4B \
  --device cuda --batch-size 4 --max-length 2048
```

The GPU script:

- uses only the 19 recommend tasks;
- uses the fixed 20-candidate pool per task;
- builds the reranker query from the real user query, active Memory preferences and hard
  requirements;
- scores title/catalogue text; it never sees evaluator labels;
- writes per-candidate Qwen3 scores and rankings, plus resources.

Then evaluate the frozen score and the untuned conservative RRF locally:

```bash
PYTHONPATH=. .venv/bin/python experiments/preference_ranking/combine_qwen3.py \
  --data runs/preference_ranking_v1/ranking_dataset.jsonl \
  --qwen3 runs/preference_ranking_v1/qwen3_scores.jsonl \
  --out runs/preference_ranking_v1/qwen3_comparison.json
```

The conservative combination is fixed before seeing results:

```text
C order = structural hard/role filter + existing scorer
Q order = same eligible set, ordered by Qwen3
E = equal-weight RRF(C, Q), k=60
```

No target-specific weights, no tuning against the 19 tasks, no labels in the model input.

Important: the prior Phase 2 Qwen3 run was **not** preference-aware and used a different
candidate pool/task set. It improved generic reranking order but is **not** a substitute for
this frozen preference-aware comparison.

## 6. Is there enough data for Ranking SFT?

**No, not from this benchmark.**

Reasons:

- Current structural/scorer baseline is already 18/19 Hit@1 and 19/19 Hit@5/Hit@10.
- Only one acceptable-but-wrong top-1 case exists (`0051`).
- Only evaluator provisional acceptable labels exist; no human pairwise preference labels
  or signed evidence comparisons.
- Unjudged candidates cannot be used as negatives.
- The frozen Qwen3 comparison is still outstanding.

Therefore:

- Do **not** build Pairwise Ranking SFT from these 19 tasks.
- Do **not** use the existing scorer output as the sole training label.
- First collect more audited hard cases where acceptable candidates are present but the
  structural order fails, with independent preference/evidence labels.

## 7. Agent E2E

No Agent E2E was run with a learned ranking layer because:

- the fixed-pool benchmark shows no meaningful headroom over the current structural
  baseline;
- only one failure exists in 19 tasks;
- no frozen reranker result is available yet.

The existing Planner Base, Memory, Gate and retrieval remain unchanged. Any future ranking
layer must be opt-in and compared against:

```text
Original BGE + current structural scorer/Gate
Original BGE + learned ranking
```

with the same Planner Base, Memory, Gate and tasks.

## 8. What is worth doing next

1. Start the AutoDL instance and run the frozen Qwen3 handoff above; report whether it fixes
   `0051` and whether it damages the other 18 tasks.
2. If it shows a real fixed-pool gain, then an opt-in ranking layer is worth prototyping;
   still no Agent adoption without E2E.
3. If it does not, keep the current Gate/scorer and move the next effort to **more audited
   hard cases**, not Planner Action SFT.
4. Preference DPO/RL remains premature: there is no human pairwise ranking set, and final
   product selection is Gate-controlled.

## 9. Artifacts

```text
runs/preference_ranking_v1/ranking_dataset.jsonl    # 30 tasks, fixed 20-candidate pools
runs/preference_ranking_v1/manifest.json
runs/preference_ranking_v1/local_summary.json
experiments/preference_ranking/build_fixed_pool.py
experiments/preference_ranking/evaluate_local.py
experiments/preference_ranking/run_qwen3_reranker.py
experiments/preference_ranking/combine_qwen3.py
```

The benchmark uses provisional evaluator labels only. It does not modify:
- `recommendation/` Scorer/Gate/Search;
- retrieval modes or indexes;
- Planner Action SFT assets;
- P18 Frozen Manifest.

The remote GPU was offline; no GPU result is reported or inferred.
