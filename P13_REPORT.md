# P13 — Modern Personalized Retrieval & Ranking Foundation 报告

## 0. 结论

**P13 可以建立独立的 Sports 实验空间，但当前可用数据只有 121 train / 18 dev / 20 reserved-test 个真实 Query–User–Target 三元组；18 个 Dev Query 太少，排序和个性化结论不具备统计可信度。检索上 BGE 略优于 BM25/BLaIR；固定预算 Hybrid 没有产生独有增量；历史个性化没有显示可靠增量。**

| 项 | 判断 |
|---|---|
| Amazon-C4 / User Purchase History 可用性 | **LIMITED：Sports 可用但覆盖严重不足** |
| Query–User–Item 监督可信度 | **LIMITED：C4 代理 Query；单 observed positive；无 judged negatives** |
| 召回模型对比 | **完成；BGE 最好但差距小** |
| 固定预算 Fusion | **无独有增量** |
| Ranking / LambdaMART | **仅探索性；数据不足** |
| 历史个性化 | **无可靠增量** |
| 代码隔离 | **PASS** |
| Agent 适配条件 | **不具备** |
| **P13 总判断** | **LIMITED** |

## 1. 数据与 Benchmark

### 1.1 数据来源

实际使用：

- `zhiyuanpeng/amazon-c4-user-purchase-history`（MemRerank release）
- `McAuley-Lab/Amazon-C4` 的 Query 文本
- 已有 Amazon Reviews 2023 Sports & Outdoors 评论，仅用于恢复目标评论 timestamp
- 现有 Sports catalogue / mappings

公开数据有 28 个类别目录，但本轮**只验证 Sports_and_Outdoors**，没有把未检查类别宣称为可用。

### 1.2 Sports 原始规模

| Split | 原始 rows | Query 文本 | Target timestamp | Target 在现有 catalogue | 保留 |
|---|---:|---:|---:|---:|---:|
| train | 342 | 342 | 319 | 144 | **121** |
| dev | 57 | 57 | 49 | 26 | **18** |
| test | 63 | 63 | 58 | 25 | **20** |
| 合计 | 462 | 462 | 426 | 195 | **159** |

说明：

- 462/462 有 Query 文本；
- 426/462 能精确恢复目标评论 timestamp（36 失败）；
- 现有 catalogue 只覆盖 195 个目标的原始行；最终 159 行同时满足 timestamp 和 catalogue；
- 其余行不伪造 retrieval 目标，直接丢弃。

`reserved-test = 20`，P13 不使用其 Gold 做选型或调参。

### 1.3 Query 与标签语义

**Query 是 Amazon-C4 由评论构造的代理 Query，不是真实搜索日志。** 因此：

- 每个 Query 只有 **1 个 observed positive**；
- 所有未标注商品是 **unjudged**，不是人工确认的负例；
- LambdaMART 中的 0 标签是 **weak/sampled negatives**；
- Hit/Recall/MRR/NDCG 只是“目标正例排名”的代理，不是完整相关性 Recall；
- 不把其他商品宣称为“用户不喜欢”。

### 1.4 时间与泄漏

- 目标 timestamp 由原始评论的 `user_id + parent_asin + normalized review text + rating` 精确匹配得到；
- 目标评论文本**只用于恢复 timestamp**，不进入历史特征；
- 所有 profile history 严格满足 `history.timestamp < target.timestamp`；
- 目标 ASIN 若出现在历史中则丢弃；
- 未使用 validation/test target 或 P6/P8/P9 Holdout；
- P11/P12 的 164 个已暴露任务仍全部为 Dev。

当前数据 split 状态：

| 检查 | 结果 |
|---|---|
| user_id 跨 split 重叠 | 0 |
| qid 跨 split 重叠 | 0 |
| target ASIN train/test 重叠 | 2 |
| dev/test target 重叠 | 0 |
| test 使用 | **reserved，不用于选型** |

### 1.5 类别决定

**本轮不切换主类别。** Sports 与现有 Sports catalogue 可直接复用；但代价是只剩 159 个有效三元组。

Electronics / Beauty & Personal Care 的 MemRerank 文件存在，但当前项目没有对应完整 catalogue / metadata / ID mapping。为避免复制旧 Sports mapping 或建立错误商品空间，P13 不贸然切换。

## 2. 数据质量限制

### 2.1 History 覆盖

159 个保留 Query 的历史规模：

| Split | rows | users | history 总条目 | 在当前 Sports catalogue | 占比 | 平均长度 |
|---|---:|---:|---:|---:|---:|---:|
| train | 121 | 121 | 19,369 | 1,342 | 6.9% | 160.1 |
| dev | 18 | 18 | 3,215 | 277 | 8.6% | 178.6 |
| test | 20 | 20 | 4,627 | 333 | 7.2% | 231.3 |

历史来自跨类别购买；只有约 7% 的历史商品在当前 Sports catalogue。  
因此 P13 的历史特征只能覆盖少量内容相似度，不能代表完整购买历史。

### 2.2 目标覆盖

- 原始目标 ASIN 在现有 Sports catalogue 的比例只有 195/462；
- 最终有效 dev 只有 18 行；
- 固定 Top-100 candidate pool 中，dev 只有 **7/18** 含 observed positive；
- 排序结果只有 7 个 dev 观测，不能作为生产选型依据。

## 3. 召回模型 Benchmark

所有模型使用：

- 同一 Query；
- 同一 Sports catalogue；
- 同一商品身份；
- 同一 observed positive；
- Top-100 定义。

### 3.1 全目录 Top-100 observed-positive 指标

| Split | Source | Hit@10 | Hit@50 | Hit@100 | MRR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| train (121) | BM25 | 14 | 32 | 42 | 0.054 | 0.069 |
| train | BGE | **21** | **48** | **66** | **0.049** | **0.077** |
| train | BLaIR | 18 | 42 | 57 | 0.044 | 0.068 |
| dev (18) | BM25 | 2 | 4 | 4 | 0.067 | 0.077 |
| dev | BGE | **2** | **6** | **6** | **0.083** | **0.091** |
| dev | BLaIR | 1 | 4 | 6 | 0.056 | 0.056 |
| reserved test (20) | BM25 | 2 | 7 | 9 | 0.057 | 0.067 |
| reserved test | BGE | **6** | **11** | **13** | **0.074** | **0.127** |
| reserved test | BLaIR | 1 | 9 | 11 | 0.025 | 0.032 |

结论：

- **BGE 在 observed-positive hit / MRR 上最好**；
- BLaIR 与 BGE 在 dev Hit@100 同为 6/18，但 BGE MRR 更高；
- BM25 最弱；
- 所有绝对数值都很低，18 dev 不足以宣称稳定胜出。

### 3.2 独有命中（Dev）

- 任一来源 hit@100：**8/18**
- 仅 BM25 命中：1
- 仅 BGE 命中：1
- 仅 BLaIR 命中：0
- 固定 50/25/25 fusion：**7/18**

### 3.3 固定预算 Fusion

固定候选预算：

```text
BM25 50 + BGE 25 + BLaIR 25
去重，确定性填充，最终最多 100
```

结果：

| Split | queries | target 在固定 pool@100 |
|---|---:|---:|
| train | 121 | 58 |
| dev | 18 | **7** |
| reserved test | 20 | 13 |

**没有证据表明固定融合产生了独有增量。** dev any-source 可达 8/18，但固定 pool 只有 7/18；融合预算反而损失了一个可达目标。由于样本只有 18，这不是稳定的量化结论，只能说明当前融合配置不值得直接进入排序层。

## 4. 排序 Benchmark

### 4.1 候选池与标签

固定候选池来自上节 fusion：

- 总预算 100；
- 每个 Query 1 个 observed positive；
- 其余候选是 unjudged，训练时标为 0，属于 **weak negatives**；
- 训练 query 只有 58 个含 positive 的组，dev 只有 7 个；
- reserved test 完全没有用于训练或调参。

因此排序结果只能算 **exploratory weak-supervision**。

### 4.2 排序基线（Dev，7 个有正例 Query）

| Ranker | Recall@10 | Recall@50 | MRR@10 | NDCG@10 | Mean target rank |
|---|---:|---:|---:|---:|---:|
| BM25 order | 0.286 | 0.571 | 0.171 | 0.198 | 45.6 |
| BGE order | **0.286** | **0.714** | **0.214** | **0.233** | **32.0** |
| BLaIR order | 0.143 | 0.714 | 0.143 | 0.143 | 43.6 |
| Pool order | 0.286 | 0.571 | 0.171 | 0.198 | 31.4 |
| Query-only LambdaMART | 0.000 | 0.714 | 0.000 | 0.000 | 34.0 |
| Query + true-history LambdaMART | 0.000 | 0.714 | 0.000 | 0.000 | 34.0 |
| Query + unrelated-history LambdaMART | 0.286 | 0.857 | 0.095 | 0.143 | 26.7 |
| Query + removed-history LambdaMART | 0.000 | 0.714 | 0.000 | 0.000 | 34.0 |

**没有证据表明训练型 ranker 优于简单的 BGE retrieval order。** 相反，7 个 Query 上的 LambdaMART 指标更差；这是样本量、单正例和弱负例共同造成的，不应解释为 ranker 方法失败，也不能作为生产结论。

### 4.3 历史个性化消融

| 配置 | Recall@10 | MRR@10 | 解释 |
|---|---:|---:|---|
| Query-only | 0.000 | 0.000 | 无历史特征 |
| True history | 0.000 | 0.000 | 与 Query-only 无差异 |
| True-history model + unrelated features | 0.286 | 0.107 | 7 Query 上的噪声，不是个性化收益 |
| True-history model + removed features | 0.286 | 0.049 | 同上 |
| Trained unrelated history | 0.286 | 0.095 | 不代表真实历史 |
| Trained removed history | 0.000 | 0.000 | 无历史信号 |

**结论：当前数据没有可靠证据表明真实历史提升 observed-positive 排名。** 不能把 unrelated 在 7 个 Query 上的波动解释为个性化成功。

### 4.4 LambdaMART 启动条件

| 条件 | 实际情况 |
|---|---|
| 训练 Query 数 | 121 总，58 含候选池正例 |
| 每个 Query 候选数 | 100（固定 pool 时） |
| Relevance 来源 | 单 observed positive + 弱负例 |
| 真实 judged negatives | 无 |
| 训练/验证隔离 | 是（train vs dev；test 保留） |
| 用户历史泄漏 | 无（时间严格早于 target） |
| **是否训练** | **是，探索性弱监督** |
| **是否可称为真实反馈 Ranker** | **否** |

## 5. 工程与代码隔离

### 5.1 目录

```text
experiments/p13/
  prepare_data.py
  target_timestamps.py
  retrieval.py
  retrieval_targets.py
  retrieval_metrics.py
  ranking.py

data/p13/sports/
  train.jsonl
  dev.jsonl
  test.jsonl
  tasks.jsonl
  manifest.json
  amazon_c4_test.csv

tests/test_p13_pipeline.py
runs/p13/{reviews,retrieval,ranking}/
```

没有复制整套旧 Serving；没有全局 Registry；没有修改默认 Serving V1、SASRec、Two-Tower、M5、Planner Contract、CandidateLedger、Role Gate、Eligibility，也没有修改 P11/P12 任务。

### 5.2 成本

| 项 | 值 |
|---|---|
| BGE catalogue index | 约 845 MB |
| BGE model | 约 877 MB |
| BLaIR model | 约 479 MB |
| BLaIR item vectors | 约 481 MB |
| BLaIR index build | GPU，约 156,746 items |
| 159-query retrieval + BLaIR index build | 约 18 分钟（RTX 4090D；含模型加载） |
| 单独 Query latency / 冷启动 | **未单独测量** |
| LightGBM | 4.7.0（本地实验环境） |

## 6. 最终回答 10 个问题

1. **Amazon-C4 历史数据真正可用的类别和规模？**
   本轮只验证 Sports：462 原始 Query（train 342 / dev 57 / test 63）；426 有目标 timestamp；现有 catalogue 只覆盖 195 个目标；最终 159 条有效（121/18/20）。其他 27 个类别目录存在，但本轮未验证。

2. **是否需要把主类别从 Sports 改到其他类别？**
   **本轮不需要也无法可靠切换。** Sports 能复用现有 catalogue，但样本只剩 18 dev；Electronics/Beauty 需要新建完整 catalogue 和 mapping，不能在 P13 内未经检查直接切换。若 P13 继续，扩展类别是必要工作，但不能假装已完成。

3. **BM25、BGE、BLaIR、Qwen3-Embedding 谁更好？**
   **BGE 在本项目真实运行中最好**：train hit@100 66 vs BM25 42 vs BLaIR 57；dev 6 vs 4 vs 6；test 13 vs 9 vs 11；dev MRR 0.083 vs 0.067 vs 0.056。差距很小且 dev 只有 18 Query。**Qwen3-Embedding 未运行**，不能给结论。

4. **混合召回是否产生真实独有增量？**
   **没有可靠证据。** 固定 50/25/25 融合 dev 7/18，任一源 8/18；BGE-only 6/18。融合没有增加可达目标，反而少 1。没有做无边界权重搜索。

5. **Qwen3-Reranker 对当前 Query 排序有没有价值？**
   **未运行，不能回答。** 当前 dev 仅 18、排序 pool 仅 7 个正例，且 Qwen 资源/时间未投入；任何价值声明都会是未经验证的。

6. **LambdaMART 是否具备合格训练条件，是否真实训练？**
   **不具备生产条件；做了探索性训练。** 58 个含正例 train Query，单 observed positive，大量 unjudged 弱负例；已训练一次固定 LightGBM LambdaRank 配置，dev 7 Query 上未优于 BGE order。不能称为真实负反馈训练的个性化 Ranker。

7. **历史个性化到底有没有增量？**
   **没有可靠增量。** Query-only 与 true-history 在 7 个 dev 上 MRR/NDCG 都为 0；历史特征没有改善 observed-positive 排名；不能把 unrelated 波动算作收益。

8. **最终推荐采用哪组召回与排序模块？**
   **LIMITED 推荐：BGE retrieval 作为后续 pilot 的召回参考；不使用当前固定 fusion、LambdaMART 或历史特征作为生产后端。** P13 没有选出可靠的生产 Retrieval/Ranking Backend。

9. **新实验路径是否与旧代码保持隔离？**
   **是。** 所有 P13 新代码在 `experiments/p13/`，数据 `data/p13/`，结果 `runs/p13/`；没有修改默认 Serving/模型/接口；P11/P12 任务未变。

10. **是否已经具备下一阶段 Agent 适配条件？**
    **不具备。** 数据只有 18 个 dev Query、7 个可用排序正例；无 judged negatives；历史覆盖 7%；个性化无增量；Qwen reranker 未验证。进入 Agent 适配前必须先扩大并修正数据/Benchmark。

## 7. GO / LIMITED / NO-GO

| 维度 | 判断 |
|---|---|
| 数据可行性 | **LIMITED** |
| 召回对比 | **LIMITED：BGE 略好，样本太小** |
| Fusion | **NO-GO（当前固定 fusion）** |
| Ranking | **NO-GO（生产）；LIMITED（探索）** |
| 个性化 | **NO-GO（无证据增量）** |
| 代码隔离 | **PASS** |
| Agent 适配 | **NO-GO** |
| **P13 总判断** | **LIMITED** |

**P13 不应直接进入 Agent 适配或 Planner SFT-v1。** 下一步必须先解决数据：

1. 扩大 Sports 或其他类别的完整 catalogue 覆盖；
2. 获取真实搜索 Query / judged relevance，或至少多正例与可靠负例；
3. 扩大 dev，建立未暴露 reserved-test；
4. 将 history 覆盖从约 7% 提升到可形成稳定用户画像；
5. 再评审 retrieval/ranking backend。

本轮未训练 Qwen Planner，未运行 SFT-v1，未修改默认 Serving V1/M5/Planner/CandidateLedger/Role Gate/Eligibility，未运行旧 Holdout，未 commit/push/deploy；GPU 实例未关闭。
