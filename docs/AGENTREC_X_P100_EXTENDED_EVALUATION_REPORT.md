# AgentRec-X — Phase 5.1: P100 Extended Benchmark & GPU End-to-End Evaluation

> 本轮新增 100 个独立 ESCI 购物 Query，构建 Agent BGE Track 与 ESCI Judged Relevance Track，实际运行 100 Query × 3 Ranker GPU 排序与 300 条 Qwen Agent E2E。
> 没有训练/替换模型；生产默认仍为 `PreferenceAwareScorer`。人类 Memory Preference Gold = 0。

---

## 0. 摘要

### P100 数据

```text
ESCI test US small_version queries: 8,956
after exclusion + electronics/query filters: 782
selected independent queries: 100
queries with exact catalog overlap: 49
judged products in Judged Track: 1,971
judged groups with valid relevance comparison: 100
Agent BGE candidate rows: 2,000
Agent exact-ESCI-judged row overlap: 14
Agent UNJUDGED candidate rows: 1,986 (99.3%)
HUMAN memory preference Gold: 0
```

### Judged Relevance Track（公开人工标签）

候选池为 **ESCI 官方已评判商品**，不是 Original BGE Top-20。
官方 gain：E=1.0, S=0.1, C=0.01, I=0。

| Model | nDCG@3 | nDCG@5 | nDCG@10 | MRR(E/S) | Pairwise Acc |
|---|---:|---:|---:|---:|---:|
| ShopRanker Query Only | **0.693587** | **0.705226** | **0.739697** | **0.943845** | **0.654720** |
| Existing Scorer | not comparable | — | — | — | — |
| Natural Memory | no memory defined; Query Only is the comparison | — | — | — | — |
| Hybrid | not defined | — | — | — | — |

nDCG@10 bootstrap 95% CI：**[0.700221, 0.780924]**。

> Existing Scorer 需要 Catalogue 结构化属性；ESCI Judged 商品不带该 schema。不得为了凑出对比而虚构属性，因此标记不可比较。

### P100 Agent Track ranker-only

| Arm | Positive Top1 Match | Negative Conflict OFF→ON | Update | Withdrawal | Top1 Change OFF→ON | Evidence Top5 |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 89.36% | 75.0% → **16.7%** | 8/8 | 9/9 | 34/100 | 482/500 |
| Natural | **93.62%** | 66.7% → **50.0%** | 8/8 | 9/9 | 35/100 | 475/500 |
| Hybrid | 61.70% | 66.7% → 41.7% | 8/8 | 9/9 | 11/100 | 469/500 |

### P100 Agent E2E

实际完成 **300 / 300 条** Agent E2E（100 tasks × 3 rankers），每 Task/Arm 独立 Executor，不复用终止状态。

| Arm | FINALIZE | ASK | Error | No Verified | Hard | Valid Top-K | Evidence | p50 | p95 | Planner p50 | Ranker p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing | **53** | 45 | 1 | 1 | 0 | 258 | 250/258 | 10.92s | 14.14s | 3.34s | n/a (CPU) |
| Natural | 50 | 48 | 1 | 1 | 0 | 247 | 237/247 | 10.80s | 14.59s | 3.26s | 0.61ms |
| Hybrid | 51 | 47 | 1 | 1 | 0 | 252 | 237/252 | 11.49s | 15.00s | 3.35s | 0.60ms |

- Top-1 与 Existing 不同的任务数：Natural **34/100**，Hybrid **39/100**；
- 300 条均无 budget_exhausted、无 hard constraint violation、无 OOM；
- 297/300 正常终态，3 条为 `p100_008` 的 Qwen Planner invalid action（同一任务三臂各 1 条）；
- 另有 3 条 `p100_063` 为 `no_verified_candidate_in_scope`（不是 FINALIZE，也不是 ASK）；
- ASK 为 **45–48%**：主要来自 `insufficient_decision_criteria` 与 `candidate_conflicts_with_stated_preference`；没有独立 Gold，不能把 ASK 叫作错误或正确；
- `FINALIZE Rate` 不是推荐准确率。

---

## 1. P100 Query Selection

- 来源：Amazon ESCI shopping_queries_dataset `small_version=1, split=test, product_locale=us`
- 官方 commit：`7916cdf6ab75a462e77f20ab40428a10923998d5`
- 排除：
  - Benchmark v2 的 100 个 base query ID
  - Phase 4.1 的 165 个 test query ID
  - 265 个 query ID 在 ID 层被排除
  - 7903 个非 Electronics 关键词 query
  - 5 个 query 长度不合格
  - 1 个有效判断不足
- 标准化文本去重后固定 Query Family。
- 选择 seed：`20261114`。
- 选择时未使用 Existing / ShopRanker / Hybrid 的任何预测结果。

结果：**100 个独立 Query**；49 个有 Catalog 精确重叠，100 个有至少一个 E/S 正例。

固定文件：

- `p100_selection_manifest.json`
- `p100_agent_tasks.jsonl`
- `p100_bge_pools.jsonl`
- `p100_judged_tasks.jsonl`
- `p100_label_coverage.json`
- `p100_memory_states.jsonl`
- `p100_candidate_metadata.jsonl`

---

## 2. 两条 Track 的候选来源

### Track A — P100 Agent Track

```text
ESCI Query
-> Original BGE Top-20 over existing 76,263-item Electronics index
-> fixed candidate IDs + metadata/evidence
-> Event-Sourced Memory Writer/Reader
-> Existing / ShopRanker Natural / Fixed Hybrid
-> Eligibility + DecisionGate + Top-K Presenter
```

- 100 个 fixed BGE Top-20 pools；
- 2,000 candidate rows；
- 1,827 unique candidates；
- BGE index 未重建；
- 所有 Ranker 使用同一候选池、同一 Eligibility 输入、同一 Memory state；
- Agent candidate pool 中只有 14 行能精确对上 ESCI 原始评判商品；其余 1,986 行为 **UNJUDGED**，从不作为负例。

### Track B — P100 Judged Relevance Track

- 每个 Query 使用 **官方 ESCI 已评判商品**；
- 100 个 Query、1,971 个 judged products；
- 所有 Query 至少有 2 个商品和 2 个相关性等级；
- 使用官方 Task 1 的 gain 映射；
- Judged Product Pool 与 Agent BGE Pool 分开保存，不混算。

---

## 3. GPU Ranker 结果

GPU 实跑：

```text
Queries: 100
Candidate pairs: 2,000 (Agent) + judged products
ShopRanker wall: 33.8 s
ShopRanker peak allocated: 6,611 MB
Query Only / Natural Memory 都实际运行
Hybrid 使用固定公式，未调权重
```

### 3.1 ShopRanker Query Only / Natural Memory

- Query Only 与 Natural 的 prompt 和 product serialization 已固定；
- serialization：`title | product_type | verified_attributes | evidence`；
- prompt hash 已保存；
- Query Only 同时供 Hybrid semantic component 复用，未重复推理。

### 3.2 Hybrid

```text
pool_z(ShopRanker Query Only)
+ 0.5 * positive_match_rate
- 0.75 * negative_conflict_rate
```

未搜索权重，未按 P100 结果调 Prompt。

---

## 4. Memory OFF / ON

100 个 Query 使用固定 Query、固定 Candidate Pool、固定 Evidence。

- OFF：空 Memory state；
- ON：预先固定的 `synthetic_event_sequence`；
- 所有 ON 状态实际经过 `ShoppingCompanionMemory.ingest_sessions` 与 `HistoryStateReader`；
- 场景分布：

```text
no_memory: 25
positive_memory: 20
negative_memory: 12
multiple_preferences: 11
preference_withdrawal: 9
preference_update: 8
query_memory_competition: 8
unknown_evidence: 7
preference_conflict: 0 (没有足够的两类互斥属性池，未强行构造)
```

State consistency：

- Update：三个 arm 都是 **8/8**；
- Withdrawal：三个 arm 都是 **9/9**；
- UNKNOWN：不是 Verified Match，也不是 Confirmed Conflict；
- Negative：OFF/ON 对比显示方向变化，但无独立人类偏好 Gold。

---

## 5. P100 Ranker-Only 详细结果

完整文件：`p100_ranker_metrics.json`、`p100_per_query_comparison.csv`

### Top-1/3/5

三个 Ranker 都完成 100/100 个 Query 排序；Top-1/3/5 候选和完整 ranking 保存在 `p100_ranker_gpu_predictions.jsonl`。

### Partial ESCI overlap（不是完整 Agent Gold）

Agent Top-20 中只有 13 个 Query 含精确 ESCI 评判商品（共 14 行）：

| Arm | Observed Relevant Hit@5（仅精确重叠） |
|---|---:|
| Existing | 9/13 |
| Natural | 6/13 |
| Hybrid | 6/13 |

这些是 **partial judged coverage**，不能解释为完整 Agent 候选池的 Hit@K；1,986 个未评判候选没有作为负例。

### Memory direction

- Positive：Natural Top1 match 93.6%，Existing 89.4%，Hybrid 61.7%；
- Negative：Existing conflict 从 75.0% 降到 16.7%，Natural 66.7%→50.0%，Hybrid 66.7%→41.7%；
- Update/Withdrawal：三者状态一致性都通过；
- 这些是结构化方向，不是人类偏好质量。

---

## 6. Agent E2E

### 6.1 规模与执行

```text
100 Agent tasks x 3 rankers = 300 trajectories
completed: 300 / 300
each task/arm: fresh Executor / fresh CandidateLedger / fresh Gate state
Qwen Planner: Qwen3.5-9B-Base, same prompt/budget
BGE: fixed P100 BGE Top-20 pool
Memory: actual ShoppingCompanionMemory + HistoryStateReader
Ranker: Existing CPU / Natural cached GPU / Hybrid cached GPU
Top-K: per-candidate eligibility + evidence validation
```

- 所有 arm 共用同一 Query、同一 fixed candidate pool、同一 Memory state、同一 Eligibility、同一 Gate 和 Action Budget。
- 没有增加 MAX_STEPS，没有强制 FINALIZE，没有跳过 Gate。

### 6.2 结果

见第 0 节表格。补充：

| Arm | Status Distribution |
|---|---|
| Existing | 53 finalized / 45 awaiting / 1 error / 1 no-verified |
| Natural | 50 / 48 / 1 / 1 |
| Hybrid | 51 / 47 / 1 / 1 |

Gate reason：

```text
evidence_sufficient:                              154
insufficient_decision_criteria:                   117
candidate_conflicts_with_stated_preference:        17
```

### 6.3 Memory 状态一致性

按实际 Writer/Reader 结果对照 100 个 ON task：

- Existing 91/100 符合预期场景；
- 9 个不一致：
  - 7 个 `unknown_evidence`：Writer 对 "verified unobtainium feature" 没有提取出有效 preference，active state 为空；
  - 2 个 `multiple_preferences`：只提取出 1 个 preference（waterproof / ergonomic）。

这是 **场景构造与 Rule-based extractor 的真实限制**，不是伪造成功；不修改 Writer/Extractor 来追分。

### 6.4 Top-K / Evidence

- 每个 finalized 结果最多 5 件，按实际合法数量返回，不补造；
- Valid Top-K items：258 / 247 / 252；
- Evidence supported：250 / 258、237 / 247、237 / 252；
- ASK 时不返回 Top-K；
- hard violations：**0**。

---

## 7. GPU / 延迟 / 工程成本

### 7.1 Ranker GPU

```text
ShopRanker P100 ranker run: 33.8 s
Peak allocated: 6,611 MB
Agent candidate rows: 2,000
Judged queries: 100
```

### 7.2 Agent E2E

```text
Qwen Planner resident: 17,551 MB
300 E2E trajectories
sum of E2E latencies: 3,329.5 s (~55.5 min)
mean E2E: 11.10 s
p50: 10.80–11.49 s
p95: 14.14–15.00 s
Planner p50: 3.26–3.35 s
Ranker p50: 0.60–0.61 ms (cached arm reorder)
OOM: 0
```

Lifecycle：ShopRanker GPU 评分先完成，再加载 Qwen 做 300 E2E；没有同时驻留 17.5GB + 6.6GB 的需求。Phase 4.4 的 Qwen+ShopRanker 20-candidate concurrent residency 测试已 PASS；本轮没有把该 smoke test 当作完整并发 E2E。

### 7.3 工程

- 不重训 Qwen / ShopRanker / DCN；
- 不重建 BGE Index；
- 不修改生产 Scorer / Gate / Planner / P18；
- Query selection 固定在前，未根据 Ranker 结果反向选择；
- 所有 data/result 文件保存 hash（见 `p100_experiment_manifest.json`）。

---

## 8. Q1–Q5 回答

### Q1. ShopRanker 是否仍然优于 Existing？

**P100 不支持“ShopRanker 明显优于 Existing”的结论。**

- Judged Relevance Track 只有 ShopRanker 可比较，Existing 因商品 schema 不可比较；
- Agent Track 没有人类 Memory Preference Gold，不能算准确率；
- Positive Top1 match：Natural 93.6% vs Existing 89.4%，差 4.2pp；
- Negative conflict：Natural ON 50.0% vs Existing 16.7%，Natural 更差；
- 13 个精确 ESCI overlap 的 partial Hit@5：Natural 6/13 vs Existing 9/13；
- 没有 Query 类别上的稳定 + 证据；
- 不能引用 P20 19/19 外推到 P100。

### Q2. Hybrid 是否真正改善 Memory？

**没有。**

- Positive Top1 match 61.7%，低于 Natural 93.6% 和 Existing 89.4%；
- Negative conflict ON 41.7%，高于 Existing 16.7%；
- Top1 OFF/ON change 仅 11%，说明结构化贡献较保守；
- Silver 同源不能当作真实偏好收益；
- 没有 Human Memory Gold。

### Q3. 100 个任务是否足以做生产决策？

- 100 个公开 judged groups；
- 100 个 Query 中只有 13 个 Agent pool 含精确 ESCI 评判；
- Agent 候选 99.3% UNJUDGED；
- 真实历史 Memory Preference Gold = 0；
- 仅机械一致性 group：全部 Memory 场景都不是人类偏好 Gold。

因此：

- **可以**支持公开相关性排序和系统稳定性比较；
- **不能**支持长期个性化质量或生产替换决策。

### Q4. 实际 Agent E2E 是否稳定？

**工程稳定，质量结论未建立。**

- 300/300 实际完成；
- 297/300 正常终态；3 条 Qwen invalid action（p100_008）；
- 0 budget_exhausted，0 hard constraint violation，0 OOM；
- Top-K Valid 247–258；Evidence 237–250；
- E2E p50 10.80–11.49s，p95 14.14–15.00s；
- Planner p50 3.26–3.35s；
- ASK 45–48%：高 ASK 主要来自证据不足/偏好冲突；没有独立 Gold，不能判定为过度澄清；
- 3 条 no_verified_candidate_in_scope（p100_063），不能算成功推荐。

结论：**流程稳定性可接受，但不能用 FINALIZE Rate 或低 hard violation 宣称推荐质量已达标。**

### Q5. 下一步是否还需要训练模型？

**不需要新增训练。**
本轮没有出现“更大 Benchmark 上 ShopRanker 明显优于 Existing 且 Memory/Gate 稳定”的证据。
推荐：

- 保留 Existing Scorer 为生产默认；
- ShopRanker Natural 继续 Shadow，可进入受控人工评测；
- Hybrid 继续 Shadow；DCN-V2 不进入生产候选；
- 下一步是人工 Memory Preference 评测，而不是继续训练模型。

---

## 9. P20 局部 Case Diff

Phase 4.3 P20 Existing success 18/19；Phase 4.5 Top-K Existing Hit@1 17/19。

Case diff 只有 `p14e2_lite_0049`：

- 两次 selected product 都是 `B08NWZ2R1K`；
- P20 success 使用 task-suite acceptable 定义；
- Phase 4.5 Top-K 使用 `acceptable_ids_in_pool` 定义；
- **不是 ranker 回归，是标签来源/指标口径差异**。

文件：`p20_hit1_metric_diff.json`。

---

## 10. 限制

- HUMAN_GOLD = 0；没有真实人工 Memory Preference 标签；
- Existing Scorer 无法在 ESCI Judged Track 上公平运行；
- Judged Pool ≠ Agent BGE Pool；
- Agent pool 99.3% UNJUDGED；
- Memory 场景是 synthetic event sequence，不是真实消费者历史；
- Preference Conflict 场景未形成（没有足够互斥属性池），不强行制造；
- Qwen Planner 的 invalid action 必须按实际错误记录，不强制 FINALIZE；
- FINALIZE Rate 不是推荐准确率；
- P100 不用于调 ShopRanker Prompt、Hybrid 权重或 Gate。
