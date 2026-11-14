# Memory Ranking Benchmark v2 & DCN-V2 Pilot Report

> Phase 4 已完成真实数据构建、ShopRanker 扩展 Shadow、DCN-V2 Silver Pilot Training。
> 生产默认 Ranker 仍为 `PreferenceAwareScorer`；未修改 Planner / Scorer / Gate / P18，未重建 BGE，未重新下载 Amazon 原始大数据，未训练 Qwen。
> 关键限制：**本轮没有 HUMAN_GOLD（=0）**。所有新 Benchmark 标签是 SILVER / MECHANISTIC / UNJUDGED；DCN-V2 结果仅是 Silver holdout，不是真实用户偏好结论。

---

## 0. 核心结果

| 项目 | 实际结果 |
|---|---:|
| 独立基础 Query | **100**（来自真实 ESCI test US shopping queries） |
| 固定 BGE Top-20 候选池 | 100 pools / **2,000 pool rows** / **1,923 unique items** |
| 总 Scenario | **300** |
| 不同 Memory 状态快照 | **37 unique states**（300 个 scenario 状态实例） |
| Pairwise 比较任务 | **300** |
| HUMAN_GOLD | **0** |
| SILVER | **145** |
| MECHANISTIC | **16** |
| UNJUDGED | **139** |
| ShopRanker P20 frozen regression | Query Only 18/19；Query+Memory 19/19（沿用上一轮冻结结果，未调 Prompt） |
| ShopRanker v2 Silver pairwise | Query Only **0.551724**；Query+Memory **0.551724**（145 pairs，接近随机） |
| ShopRanker v2 Top-1 变化（Query→Memory） | **89 / 300** |
| DCN-V2 训练数据 | 145 groups / 290 candidate rows / 98 base queries |
| DCN-V2 Memory OFF test | 13/14 = **0.928571** |
| DCN-V2 Memory ON test | 14/14 = **1.000000** |
| DCN-V2 训练 GPU | **<1 s / arm，peak ~18 MB** |

**结论：** Benchmark v2 成功建立，并且比 P20 更难、更能暴露 Memory 行为问题。
ShopRanker 在 P20 上 19/19，但在新 Silver pairwise 上只有 **55.17%**；加入 Memory 后总体 pairwise 不变，说明 P20 的 19/19 不能推广成“可靠 Memory 排序”。
DCN-V2 ON 在 Silver holdout 上 14/14 是**规则标签上的可预期结果**，不是用户偏好质量。

---

## 1. 真实构建的独立 Query 数量

### 1.1 来源

- 真实 ESCI test，`small_version=1`，`product_locale=us`。
- 选择 **至少有一个 exact catalog overlap product** 的 query，共 237 个候选。
- 固定 seed 42，随机抽取 **100 个独立基础 Query**。
- 每个 Query 只生成一个 BGE Top-20 candidate pool；同一 Query 的多个 Memory 状态共享该 pool，不重复计算独立 Query。

### 1.2 数量

```text
independent_base_queries: 100
base_query_splits: 68 train / 21 val / 11 test
```

- 所有 100 个 Query 都至少有一个 ESCI exact catalog overlap label。
- ESCI 原始 human relevance 仅作为候选来源/外部标签资产保留，**没有**被当成长期 Memory Gold。
- 没有把 P20 Query 混入新模型训练；P20 仅做冻结 regression。

---

## 2. Memory 状态数与场景分布

### 2.1 规模

| 指标 | 数值 |
|---|---:|
| 独立基础 Query | 100 |
| Scenario 实例 | **300** |
| 不同 Memory 状态快照 | **37** |
| 每个 Query 平均 Scenario | 3.0 |
| 同一 Query 最多 Memory 变体 | 3 |

同一 Query 的多个 Memory 状态共享固定 candidate pool；**不把 Memory 变体当作完全独立 Query**。

### 2.2 Scenario 类型

| Scenario Type | 数量 | 说明 |
|---|---:|---|
| no_memory | 100 | 无历史偏好 |
| positive_memory | 100 | 正向 synthetic preference |
| negative_memory | 17 | 负向 synthetic preference |
| query_memory_competition | 16 | Query 与 Memory 竞争 |
| preference_update | 17 | 正偏好被更新 |
| preference_withdrawal | 17 | 正偏好被撤回 |
| unknown_evidence | 17 | verified 与 UNKNOWN 比较 |
| preference_conflict | 16 | 冲突，必须 ASK |
| **Total** | **300** | — |

### 2.3 Memory 来源标记

- 本轮所有历史偏好均为 **`synthetic_preference_scenario`**。
- **`recorded_event_sourced_memory` = 0**；没有把 P20 的真实 Event-Sourced 状态混入新 Benchmark。
- 这样避免把 P20 的 provisional label / 已调优历史当成独立 Gold。

### 2.4 Query–Memory 关系

```text
none: 134
memory_only: 135
complement: 15
competition: 16
```

---

## 3. 真实候选商品与商品对

| 指标 | 数值 |
|---|---:|
| Candidate pool rows | 2,000 |
| Unique candidate items | **1,923** |
| 商品 pairs | **300** |
| Candidate with metadata | 1,885 / 1,923 |
| Metadata missing | 38 |
| Candidate pool | Original BGE Top-20 |
| BGE index | 现有 76,918 × 768，不重建 |

- 每个 pair 使用真实 Amazon Electronics 商品、title、category、verified attributes、evidence，以及 UNKNOWN 状态。
- 不是只挑“特别容易区分”的商品；pair 选择覆盖：
  - positive match vs unknown；
  - negative conflict vs explicit safe alternative；
  - update / withdrawal 同一商品族；
  - conflict 两件商品分别满足互斥偏好；
  - query-memory competition；
  - UNKNOWN evidence。
- **Eligibility**：本 Benchmark 的 synthetic query 没有声明 hard requirement，因此 20 个候选全部 eligible；hard_requirements=0，hard_violations=0。这与 P20 的 shared eligible set 不同，报告中不混用。

---

## 4. Gold / Silver / Mechanistic / Unjudged 数量

| Provenance | 数量 | 含义 |
|---|---:|---|
| **HUMAN_GOLD** | **0** | 没有真实人工评审 |
| EXTERNAL_RELEVANCE | **0** | pair 中没有形成可用的 ESCI exact 双候选标签；ESCI 仅作为 Query 来源 |
| **SILVER** | **145** | 基于 verified attribute evidence + 明确规则 |
| **MECHANISTIC** | **16** | conflict / 状态正确性测试，不是偏好 Gold |
| **UNJUDGED** | **139** | 证据不足或没有可靠偏好判断 |

### 4.1 SILVER 生成规则

- Positive evidence verified vs unknown：verified 候选为优选；
- Negative conflict vs explicit safe alternative：safe 候选为优选；
- Update 后的新偏好：新偏好 verified 候选为优选；
- 只使用商品属性和明确规则；
- 不使用 Scorer 最终分数、ShopRanker 分数、BGE 排名或 DCN 预测生成标签；
- 候选 A/B 顺序交替，避免“永远 A 赢”的标签/位置偏置。

### 4.2 重要警告

Existing Scorer 在 145 个 SILVER pairs 上为 **145/145 = 1.0**。
这是因为 SILVER 规则与 Scorer 的属性匹配逻辑同源，**不是独立质量证据**。本报告不会把这个 1.0 作为“Existing Scorer 优于 ShopRanker”的结论。

---

## 5. 可直接人工标注的数据包

路径：`data/memory_ranking_benchmark_v2/`

```text
annotation_tasks.jsonl      300 tasks
annotation_template.csv     300 rows
annotation_guidelines.md    guidelines
```

Annotation package 隐藏：

- BGE rank / score
- Existing Scorer score
- ShopRanker score
- DCN-V2 score
- 生成的 SILVER / MECHANISTIC / UNJUDGED 标签
- 规则预期答案

已验证 `annotation_tasks.jsonl` 中不含 `bge_rank`、`bge_score`、`shopranker`、`dcn`、`final_score`、`label_provenance` 等字段。
人工标签选项：`A / B / BOTH / INSUFFICIENT`；conflict 必须 `INSUFFICIENT`，不能强行 A/B。

---

## 6. ShopRanker 扩展 Shadow

### 6.1 协议

- Model: `srpone/zoowork-shopranker-0.6b`，Frozen，官方 prompt，官方 `yes/no` logit。
- 300 scenarios × 20 candidates。
- B: Query Only。
- C: Query + synthetic Active Memory（作为官方 Query 后缀）。
- 同一 candidate pool、同一 metadata、同一 memory snapshot。
- Top-20 ranking 与 raw scores 已保存。
- 没有用 SILVER 或 P20 调 Prompt / 权重。

### 6.2 Silver pairwise（145 pairs）

| Arm | Pairwise Accuracy | Correct / N | Mean rank gap |
|---|---:|---:|---:|
| Existing Scorer (tautological) | 1.000000 | 145/145 | 6.7103 |
| ShopRanker Query Only | **0.551724** | 80/145 | 6.7448 |
| ShopRanker Query + Memory | **0.551724** | 80/145 | 7.6069 |
| DCN-V2 Memory ON (test subset) | 1.000000 | 14/14 | — |

**ShopRanker Query Only 与 Query+Memory 的 Silver pairwise 完全相同**；Memory 没有带来总体提升。

### 6.3 按 Scenario 类型

| Scenario | n Silver | Shop Query | Shop Memory |
|---|---:|---:|---:|
| positive_memory | 97 | 0.567010 | 0.536082 |
| unknown_evidence | 16 | 0.437500 | 0.562500 |
| preference_update | 17 | 0.529412 | 0.588235 |
| query_memory_competition | 15 | 0.600000 | 0.600000 |

- Positive memory 上 Memory 版本反而略低；
- UNKNOWN 和 UPDATE 上 Memory 版本略高，但样本只有 16/17，不能宣称可靠；
- 总体 145 pairs 不变。

### 6.4 Top-1 / Top-3 变化

```text
ShopRanker Query Only -> Query + Memory:
  Top-1 changed: 89 / 300
  Top-3 set changed: 148 / 300
  Full ranking changed: 183 / 300

Existing Scorer -> ShopRanker Query+Memory:
  Top-1 changed: 256 / 300
  Top-3 set changed: 296 / 300
```

ShopRanker 对 Memory 文本高度敏感，但变化并不对应 Silver 偏好。
**“排序发生了变化” ≠ “Memory 正确”。**

### 6.5 Negative / Update / Withdrawal 行为

**Negative Memory 方向一致性（mechanistic，不是 Gold）：**

```text
17 negative scenarios:
  Existing Scorer safe-above-conflict: 17/17 (tautological)
  ShopRanker Memory ON safe-above-conflict: 8/17
```

ShopRanker 在 negative memory 上接近随机/反向，不能声称理解 negative preference。

**Update / Withdrawal 前后状态：**

```text
34 state-transition scenarios:
  full ranking changed: 34/34
  top-1 changed: 19/34
  top-3 set changed: 31/34
  pair order changed: 8/34

Update:
  top-1 changed 6/17, pair order changed 4/17
Withdrawal:
  top-1 changed 13/17, pair order changed 4/17
```

状态变化几乎总会改变排序，但**没有独立标签判断方向是否正确**；只能说是 mechanistic response，不是 consistency success。

### 6.6 Verified vs UNKNOWN

- `unknown_evidence` 16 SILVER pairs：
  - Shop Query：7/16 = 0.4375
  - Shop Memory：9/16 = 0.5625
- 提升存在，但样本极小；不能证明 UNKNOWN 推理能力。

### 6.7 Hard 约束、延迟、显存

```text
hard_requirements: 0
hard_constraint_violations: 0

ShopRanker Query+Memory, 20 candidates, batch call:
  mean 116.728 ms
  p50 110.472 ms
  max 234.563 ms
Peak GPU: 5,136.52 MB

Existing Scorer CPU:
  mean 0.095 ms
  p50 0.092 ms
```

### 6.8 P20 冻结 regression

沿用上一轮冻结结果，**没有重新调 Prompt**：

| P20 Arm | Hit@1 |
|---|---:|
| Existing Scorer Memory ON | 18/19 |
| ShopRanker Query Only | 18/19 |
| ShopRanker Query+Memory | 19/19 |
| Qwen3-Reranker-0.6B Base+Memory | 18/19 |

P20 的 19/19 不进入新 Benchmark 的 Gold，也不作为泛化证据。

---

## 7. DCN-V2 实际训练数据规模

路径：`data/memory_ranking_benchmark_v2/dcn_v2_pairs/`

| 指标 | 数值 |
|---|---:|
| Groups | **145** |
| Candidate rows | **290** |
| Base queries | **98** |
| Features | **113** |
| Memory features | **55** |
| Memory-OFF features | **58** |
| Train groups | **97** |
| Val groups | **34** |
| Test groups | **14** |
| Label provenance | **SILVER only** |
| HUMAN_GOLD | **0** |
| UNJUDGED as negatives | **No** |

Split：

```text
train: 97 groups / 194 rows
val:    34 groups /  68 rows
test:   14 groups /  28 rows
```

Base-query-family split：同一个 base query 的所有 Memory 状态只在一个 split；不同 split 无 Query 家族泄漏。
本轮实际 split：base queries 68 train / 21 val / 11 test；scenarios 204 / 63 / 33；DCN groups 97 / 34 / 14。
不把 P20 provisional acceptable 当作训练 Gold。
不把 UNJUDGED 商品当作负例。

---

## 8. DCN-V2 Pilot 训练结果

模型：现有 `experiments/dcn_v2/model.py` 的 DCN-V2 / DCN-Mix 小型实现；固定参数，无大型超参数搜索。训练 seed=42。

```text
hidden = 64
cross_layers = 3
dropout = 0.1
optimizer = AdamW, lr=1e-3, weight_decay=1e-4
loss = pairwise BCE(score_A - score_B, label_A)
max epochs = 60, early stopping patience = 12
batch groups = 32
GPU = RTX 4090 D
```

### 8.1 Memory OFF

| Metric | Value |
|---|---:|
| Input features | 58 |
| Parameters | 32,705 |
| Train groups | 97 |
| Val groups | 34 |
| Test groups | 14 |
| Best epoch | 1 |
| Epochs ran | 13 |
| Train loss last | 0.053523 |
| Train accuracy | 0.927835 |
| Val accuracy | 0.882353 |
| **Test accuracy (Silver)** | **0.928571 (13/14)** |
| Train time | **0.609 s** |
| Peak GPU | **18.14 MB** |

### 8.2 Memory ON

| Metric | Value |
|---|---:|
| Input features | 113 |
| Parameters | 39,745 |
| Train groups | 97 |
| Val groups | 34 |
| Test groups | 14 |
| Best epoch | 1 |
| Epochs ran | 13 |
| Train loss last | 0.002361 |
| Train accuracy | 1.000000 |
| Val accuracy | 1.000000 |
| **Test accuracy (Silver)** | **1.000000 (14/14)** |
| Train time | **0.344 s** |
| Peak GPU | **18.29 MB** |

### 8.3 必须明确的解释

DCN-V2 ON 的 14/14 **不是**“模型学会了真实用户偏好”。
SILVER 标签由属性匹配规则生成，而 Memory ON 输入直接包含同样的 match/conflict 特征；模型在 1 个 epoch 内达到 train/val 1.0，属于**规则可学习性 + 小样本过拟合**。
Holdout 只有 14 groups；**不能把 1.0 当成用户偏好质量**。

---

## 9. Memory OFF / ON 对照与最终比较

### 9.1 New Benchmark Silver Pairwise

| Model | Test set | Pairwise | Correct/N | Provenance |
|---|---|---:|---:|---|
| Existing Scorer | all 145 Silver | 1.000000 | 145/145 | tautological Silver |
| ShopRanker Query Only | all 145 Silver | 0.551724 | 80/145 | Silver |
| ShopRanker Query + Memory | all 145 Silver | 0.551724 | 80/145 | Silver |
| ShopRanker Query Only | DCN test 14 | 0.357143 | 5/14 | Silver |
| ShopRanker Query + Memory | DCN test 14 | 0.500000 | 7/14 | Silver |
| DCN-V2 Memory OFF | DCN test 14 | **0.928571** | 13/14 | Silver |
| DCN-V2 Memory ON | DCN test 14 | **1.000000** | 14/14 | Silver |
| HUMAN_GOLD | — | — | 0 | **无** |

### 9.2 P20 Provisional Regression（单独表，不混分）

| Model | P20 Hit@1 |
|---|---:|
| Existing Scorer Memory ON | 18/19 |
| ShopRanker Query Only | 18/19 |
| ShopRanker Query + Memory | 19/19 |
| DCN-V2 | 未在 P20 训练/评测 |

### 9.3 Mechanistic / Counterfactual

```text
16 conflict scenarios: requires_clarification
  -> ASK, no forced Top-K Gold

Negative memory directional:
  Existing 17/17; ShopRanker Memory 8/17

Update/Withdrawal:
  34 state transitions; full ranking changed 34/34, top-1 19/34,
  pair order changed 8/34; no independent direction labels
```

DCN-V2 的 counterfactual consistency **未评估**：当前 DCN 数据只包含 after-state Silver pairs，没有 before/after 状态监督；不伪造该指标。

---

## 10. 两种新排序方案的不足与下一步

### ShopRanker-0.6B

不足：

1. P20 19/19 不能推广；Benchmark v2 Silver 只有 55.17%，接近随机。
2. Query Only 与 Query+Memory 总体 pairwise 相同；Memory 没有可靠提升。
3. Negative memory 方向仅 8/17；不能声称理解负向偏好。
4. Top-1 变化 89/300，说明对文本敏感但不代表语义正确。
5. UNKNOWN / update 的小幅提升样本只有 16/17，不显著。
6. 没有真实用户 Gold；只有 SILVER。
7. Peak GPU 5.14 GB，单 scenario 20 candidates 约 117 ms，作为生产 ranker 成本可接受但不是当前瓶颈。

### DCN-V2

不足：

1. 只有 145 Silver groups / 290 rows；98 个 base queries；test 仅 14 groups。
2. 模型 1 epoch 达到 1.0，明显是规则标签可学习/过拟合。
3. Memory ON 特征包含生成 SILVER 的 match/conflict 信息，存在规则同源性。
4. 没有 HUMAN_GOLD；没有真实曝光/点击/选择监督。
5. 未评估 counterfactual consistency。
6. 不能以 14/14 宣称真实用户偏好质量。

### 下一步（按优先级）

1. **人工评审 annotation package**：300 tasks；目标至少 200 条 HUMAN_GOLD。
2. 用真实 Event-Sourced Memory 状态替换/补充 synthetic 场景，并保持 `recorded_event_sourced_memory` 标记。
3. 增加基础 Query 到 150–300，独立人工 pairwise 标签至少 1,000 对。
4. DCN-V2 在 HUMAN_GOLD / 明确 SILVER 上重训；使用独立 test 和真实曝光候选。
5. ShopRanker 只作为 zero-shot baseline；不在 P20 上调 Prompt。
6. 不恢复旧 LightGBM / Neural Memory 模型。

---

## 11. 可复现文件、Seed、Split、模型版本

### 11.1 Benchmark v2

```text
data/memory_ranking_benchmark_v2/
  base_queries.jsonl              100 ESCI-derived queries
  bge_pools.jsonl                 100 x Top-20 fixed pools + BGE scores
  candidate_metadata.jsonl        1,923 real candidates
  scenarios.jsonl                 300 scenarios
  pair_tasks.jsonl                300 pair tasks + labels/provenance
  annotation_tasks.jsonl          300 hidden-label annotation tasks
  annotation_template.csv         300-row template
  annotation_guidelines.md
  manifest.json
```

Seed：42（query selection）；BGE retrieval 无随机性。
Query selection 237 eligible -> random sample 100；所有 Memory variants 共享 base query split。

### 11.2 DCN-V2 data

```text
data/memory_ranking_benchmark_v2/dcn_v2_pairs/
  pairs.parquet
  feature_schema.json
  manifest.json
```

### 11.3 运行结果

```text
runs/memory_ranking_benchmark_v2/
  shopranker_shadow.json          300 scenarios, Top-20 raw scores
  existing_scorer_shadow.json
  state_shadow.json               34 update/withdraw state transitions
  shadow_metrics.json             145 Silver pair metrics
  dcn_v2_pilot.json               2 arms, train/val/test
  dcn_v2_off.pt
  dcn_v2_on.pt
```

模型版本：

```text
ShopRanker: srpone/zoowork-shopranker-0.6b (frozen; Apache-2.0)
Existing Scorer: preference_scorer_v0.1
DCN-V2: local experiments/dcn_v2/model.py; hidden64, cross3
BGE: BAAI/bge-base-en-v1.5 existing 76,918 x 768 index (not rebuilt)
```

### 11.4 代码

```text
experiments/memory_ranking_v2/build_benchmark.py
experiments/memory_ranking_v2/shopranker_shadow.py
experiments/memory_ranking_v2/existing_scorer_shadow.py
experiments/memory_ranking_v2/state_shadow.py
experiments/memory_ranking_v2/evaluate_shadow.py
experiments/dcn_v2/build_benchmark_v2_pairs.py
experiments/dcn_v2/train_benchmark_v2_pilot.py
```

---

## 12. 最终交付判断

- **Benchmark v2**：实际构建 100 个独立 Query、300 个 scenario、300 个 pair 任务、1,923 个真实候选商品；不是设计文档。
- **Annotation package**：300 个可直接人工评审的任务；HUMAN_GOLD=0，未伪造。
- **ShopRanker 扩展 Shadow**：300 scenarios 全部真实运行；P20 19/19 没有迁移到新 benchmark；新 Silver pairwise 55.17%，Memory 总体无提升。
- **DCN-V2 Pilot**：完成真实 Silver 训练；145 groups；Memory OFF/ON test 0.929/1.000；但这是规则标签结果，不是用户偏好质量。
- **生产**：仍为 `PreferenceAwareScorer`；所有 Learned Ranker 仅 Shadow，未替换生产。
- **P20**：仅冻结 regression，不参与新模型训练与调参。

**Phase 4 完成标准：** Benchmark v2 可复用数据包存在；ShopRanker 扩展 Shadow 完成；DCN-V2 有明确标签来源（SILVER）的 Pilot 训练结果；HUMAN_GOLD 为 0 的事实已明确记录。
