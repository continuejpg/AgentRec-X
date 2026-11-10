# P13-R — Personalized Search Benchmark Rebase 报告

## 0. 结论

**Electronics 是明显比 Sports 更合适的新底座：严格有效样本 1,095（Train 817 / Dev 135 / Reserved Test 143），最终隔离后 Train 817 / Dev 123 / Test 129；目标商品 metadata 覆盖 1,095/1,096，目标 timestamp 覆盖 1,095/1,096。BGE 在真实运行中显著优于 BM25。**

但 P13-R 仍不是 GO：

- 固定 BM25+BGE Top-100 pool 只覆盖 **40/123 Dev** 正例；
- 只有单 observed positive，没有 judged negative；
- 历史文本 metadata 只有约 **6%**，无法形成可靠个性化监督；
- 固定历史规则使排序变差，没有个性化增量；
- 按停止条件未训练新的 LambdaMART。

| 维度 | 判断 |
|---|---|
| 类别/数据迁移 | **GO：Electronics 数据规模明显优于 Sports** |
| 商品目录 target 覆盖 | **GO：1,095/1,096** |
| BM25/BGE 检索基线 | **LIMITED：BGE 有真实信号，固定融合仍需改进** |
| Dev 规模 | **GO（123）用于检索诊断；LIMITED 用于正式排序** |
| 历史个性化监督 | **NO-GO：无可靠证据** |
| 新 Ranker | **NO-GO（当前阶段）** |
| Agent 适配 | **NO-GO** |
| **P13-R 总判断** | **LIMITED** |

## 1. 类别可行性对照

### 1.1 Electronics vs Beauty

| 类别 | 原始 train/dev/test | 严格有效 train/dev/test | Target metadata | Target timestamp |
|---|---:|---:|---:|---:|
| **Electronics** | 818 / 135 / 143 | **817 / 135 / 143** | 818/818 等 | 817/818 等 |
| Beauty_and_Personal_Care | 662 / 120 / 114 | 662 / 120 / 114 | 全部 | 全部 |

- 两类的 C4 Metadata 覆盖率都高；
- Electronics 的严格样本更多（1,095 vs 896），Dev 也更多（135 vs 120）；
- 选择 **Electronics** 作为主类别；
- **Beauty 作为候补**，本轮不继续跑 Beauty 的完整索引。

### 1.2 Electronics 最终 split

原始 MemRerank split 有轻微用户/目标重叠。P13-R 用确定性规则清理：

> 删除 dev/test 中 user 或 target 同时出现在更早 split 的行；train 不变。

| Split | 原始 | 清理后 | 说明 |
|---|---:|---:|---|
| train | 817 | **817** | 不变 |
| dev | 135 | **123** | 删除 12 个重叠行 |
| test | 143 | **129** | 只作为 reserved，不运行 |
| 合计 | 1,095 | **1,069** | |

清理后：

```text
train/dev/test user overlap = 0
train/dev/test qid overlap = 0
train/dev/test target overlap = 0
```

`test` 没有运行、没有看逐条 ranking、没有用于选型。

## 2. 商品目录与防泄漏

### 2.1 新 catalogue

- 来源：`Amazon-C4 sampled_item_metadata_1M` 的 **Electronics subset**；
- 身份：`item_id` / `parent_asin`；
- 目录大小：**76,918**；
- 不使用旧 Sports 5-core `item2id`；
- 不再因为商品“不在旧 mapping”而删除；
- 但必须诚实说明：这是 **C4 sampled metadata catalogue**，**不是完整 Electronics 全目录**；不能声称全量 catalogue。

target 覆盖：

- Electronics 原始 1,096 个 target；
- 1,095 个有严格有效 timestamp；
- **1,095/1,096（99.9%）** 的 target 在该 catalogue 中；
- Sports 旧路线的 target 丢失问题基本修复。

### 2.2 时间与目标

- 目标 timestamp 来自 Amazon Reviews 2023 的官方 timestamp benchmark CSV；
- 1,095/1,096 恢复成功；
- 所有 history 严格 `timestamp < target timestamp`；
- target 商品若出现在 history 则丢弃；
- target review 原文没有作为历史偏好或排序特征；
- C4 Query 由 target review 生成，标识为 **proxy query**；**原始搜索 Query 与独立改写版本不可得**，C4 文本本身是改写后的代理 Query；
- 1,069 个最终任务、1,068 个用户；Query 文本**无 exact duplicate**（0 重复）；只有 21 个 target ASIN 被多个用户共享，因此“同 Query 不同用户”的真实对照并不存在；
- 同用户多 Query、跨 target 的 personalized 对照也有限，不能把 C4 proxy Query 当成搜索日志；
- 未触碰 P6/P8/P9 Holdout、P11/P12 原始任务。

### 2.3 History 覆盖

| Split | queries | history items | 有文本 metadata | 仅 identity/category |
|---|---:|---:|---:|---:|
| train | 817 | 77,551 | 4,441（5.7%） | 73,110 |
| dev | 123 | 9,575 | 591（6.2%） | 8,984 |

History 跨类别保留；能恢复文本和只能恢复交互身份的比例被单独记录。  
**不能把大量 identity-only 历史当成文本偏好证据。**

## 3. BM25 / BGE 检索

同 Query、同 catalogue、同 observed positive、同 Top-100 定义。

| Split | Source | Hit@10 | Hit@50 | Hit@100 | MRR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| train (817) | BM25 | 44 | 88 | 114 | 0.027 | 0.033 |
| train | **BGE** | **147** | **266** | **352** | **0.086** | **0.108** |
| dev (123) | BM25 | 6 | 14 | 17 | 0.018 | 0.025 |
| dev | **BGE** | **15** | **36** | **53** | **0.062** | **0.076** |

### 3.1 独有命中

| Split | BGE 独有 | BM25 独有 |
|---|---:|---:|
| train | 256 | 18 |
| dev | **40** | **4** |

**BGE 明显强于 BM25**；BM25 只提供少量独有目标。  
因此本轮没有继续跑 BLaIR/Qwen3-Embedding——不是强行追新模型，而是先用 BM25+BGE 判断数据质量。

### 3.2 固定候选池

固定预算：

```text
BM25 Top-50 + BGE Top-50
去重、确定性填充，最多 100
```

| Split | queries | target in pool@100 |
|---|---:|---:|
| train | 817 | 285（34.9%） |
| dev | 123 | **40（32.5%）** |

注意：BGE 单路 Top-100 可命中 **53/123 Dev**，固定 50/50 pool 只有 **40/123**。  
说明固定融合配额确实损失了 BGE 可达目标；但 P13-R 的目标是判断数据基础，不是继续无界调融合参数。

## 4. 历史有效性诊断

固定候选池上，用固定规则比较 Query-only 与 Query+History，不做训练：

- Query score = BM25/BGE RRF 排名分数；
- History score = Query score + 历史类别匹配 + 文本 overlap + 时间衰减；
- 没有根据 Dev 正例调权重。

直接结果：

| Split | 配置 | Recall@10 | MRR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| train | Query-only | 0.368 | 0.158 | 0.207 |
| train | Query+History | 0.140 | 0.053 | 0.073 |
| dev | Query-only | 0.350 | **0.153** | **0.199** |
| dev | Query+History | 0.150 | 0.029 | 0.057 |

- 历史规则在 dev/train 上**都变差**；
- 有 28/40 dev 和 212/285 train 的目标排名发生变化，但这只是排序变化，不是质量提升；
- 历史文本覆盖只有约 6%，大多只能做 identity/category；
- **没有可信证据支持历史个性化有效。**

## 5. 回答 7 个问题

1. **哪个类别适合作为新排序底座？**
   **Electronics。** 严格有效 1,095 个（817/135/143），比 Beauty 896 更大；价格/类目和 C4 metadata 可用；Beauty 保留为候补。

2. **是否修复了 Sports target 丢失？**
   **是。** Sports 有效目标只有 159/462；Electronics target metadata/timestamp 覆盖 **1,095/1,096（99.9%）**。但 catalogue 是 C4 sampled Electronics subset（76,918），不是完整 Electronics 全目录，仍需在正式训练前补全。

3. **Dev 规模和候选可达率是否足以支持正式排序实验？**
   **用于检索诊断足够；用于正式个性化排序仍不足。** Dev 123，BGE hit@100 53，固定 pool 40；单 observed positive、无 judged negative，不能支撑可信 ranker 结论。

4. **是否存在可信的历史个性化监督？**
   **没有。** history 文本 metadata 约 6%；没有 judged negatives；固定历史规则变差；不能把单正例 + 大量 unjudged 候选称为真实点击反馈学习。

5. **应继续 LambdaMART、先补人工相关性标签，还是转向其他数据集？**
   **先补人工相关性标签/judged negatives；不要现在训练 LambdaMART。**
   如果目标是真实个性化搜索而非 C4 代理 Query，下一步应评估 **JDsearch 等真实搜索日志**，而不是继续用 C4 单正例堆 Ranker。

6. **是否值得进一步开发新 Ranker？**
   **当前 NO-GO。** BGE retrieval 已有可用信号；但在没有可靠负例/多正例/相关性判断之前，新 Ranker 会产生不可解释的代理结果。

7. **是否保持与旧实现隔离？**
   **是。** 新数据/代码/结果在 `data/p13r/`、`experiments/p13r/`、`runs/p13r/`；未修改默认 Serving、SASRec、Two-Tower、M5、Planner、CandidateLedger、Role Gate、Eligibility；P13 原始负结果和报告保留；P11/P12 任务未变。

## 6. 工程与资源

| 项 | 值 |
|---|---|
| Catalogue | 76,918 Electronics items |
| BGE 索引 | 236,292,096 bytes（约 225 MiB） |
| BGE 索引构建 | 约 128 秒（RTX 4090D） |
| 检索任务 | 940（train 817 + dev 123） |
| 检索耗时 | 约 9 分钟（含索引加载/BGE 编码，未单独拆 quer latency） |
| 新模型训练 | **无** |
| Qwen/BLaIR | 未运行；BM25+BGE 已足够判定数据质量 |
| 测试 | P13-R 3 项 + 编译 |

## 7. GO / LIMITED / NO-GO

| 项 | 状态 |
|---|---|
| 类别迁移与 target 覆盖 | **GO** |
| 时间切分/隔离 | **GO** |
| BM25/BGE retrieval 基线 | **LIMITED：BGE 有信号，融合损失正例** |
| 历史个性化 | **NO-GO（无增量）** |
| 新 LambdaMART / Ranker | **NO-GO（当前无可靠监督）** |
| Agent 适配 | **NO-GO** |
| **P13-R 总判断** | **LIMITED** |

**建议：**

1. 保留 Electronics + BGE retrieval 作为新底座；
2. 不继续无边界尝试 LambdaMART；
3. 先补人工相关性标签、多正例和 judged negatives；
4. 如果目标是真实搜索个性化，优先评估 JDsearch 等真实搜索日志；
5. 补全 Electronics full metadata catalogue 后再决定是否进入 Agent。

本轮未训练新 Ranker、未训练 Planner、未运行 SFT-v1、未运行 reserved-test 或旧 Holdout、未修改旧 Serving；未 commit/push/deploy；GPU 实例未关闭。
