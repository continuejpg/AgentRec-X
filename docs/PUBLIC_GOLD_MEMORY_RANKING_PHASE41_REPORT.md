# Public-Gold-Driven Preference-Conditioned Ranking — Phase 4.1 Report

> 本轮完成真实 ESCI 官方人工标签训练、三组 DCN-V2 对照、ShopRank-Bench / PersonaShop 公开评测、以及 AgentRec-X Shadow。
> 关键边界：ESCI 是 Query–Product 相关性监督；从 Query 拆出的偏好是 **query_derived_proxy**，不是真实长期 Event-Sourced Memory。ShopRank-Bench / PersonaShop 也不是 HUMAN_GOLD。**本轮 HUMAN_GOLD = 0。**

---

## 0. 结果总览

### Level 1 — Public Relevance Gold（ESCI）

| Arm | nDCG@3 | nDCG@5 | nDCG@10 | MRR | Pairwise |
|---|---:|---:|---:|---:|---:|
| A. Base Query Only | 0.767278 | 0.790064 | 0.916968 | 0.654863 | 0.582875 |
| B. Base Query + Structured Preference | **0.797396** | **0.806033** | **0.927252** | **0.720003** | 0.584753 |
| C. Original Full Query | 0.780677 | 0.801618 | 0.919585 | 0.671316 | **0.588380** |

- **B − A**：nDCG@10 **+0.010284**，MRR **+0.065140**，pairwise **+0.001878**。
- **C − A**：nDCG@10 **+0.002617**，MRR **+0.016453**，pairwise **+0.005505**。
- **B − C**：nDCG@10 **+0.007667**，MRR **+0.048687**，pairwise **−0.003627**。

结论：结构化偏好条件在 ESCI 官方相关性任务上提高了 nDCG/MRR；full query 的 pairwise 略好，但 structured preference 的 ranking/MRR 更好。

### Level 2 — Preference-Conditioned Proxy

**成立但只能算代理**：B 优于 A，说明“把当前 Query 中拆出的属性显式结构化”有助于恢复 ESCI 相关性判断。
**不能**据此声称真实长期 Memory 有因果收益；偏好来自当前 Query，不是历史记忆。

### Level 3 — Public Judge / Profile Preference

| Dataset / Model | Result |
|---|---:|
| ShopRanker-0.6B on ShopRank-Bench preference（10,511 pairs） | **76.5294%** |
| ShopRanker-0.6B on AHP（1,500） | **31.6667%** |
| ShopRanker-0.6B on Budget（1,302） | **79.8003%** |
| Qwen3-Reranker-0.6B base on 1,000 preference subset | **73.6%** |
| PersonaShop profile score shift | 90% examples lower with profile; **no accuracy claim** |

### Level 4 — Actual Event-Sourced Memory (AgentRec-X Shadow)

| Arm | P20 Hit@1 | Benchmark v2 Silver pairwise | 0051 |
|---|---:|---:|---|
| DCN A Base Query Only | **16/19** | **0.689655** | acceptable top1 |
| DCN B Preference ON | **13/19** | **0.262069** | acceptable rank 3 |
| DCN C Full Query | **15/19** | **0.703448** | acceptable rank 2 |
| Existing Scorer | 18/19 | 1.000* | 18/19 |
| ShopRanker Query+Memory | 19/19 | .551724 | top1 |

\* Existing Scorer 的 Silver 1.0 是规则同源，不是独立质量。

**DCN-V2 的 ESCI 收益没有迁移到 Agent Memory**；B Preference ON 在 P20 反而从 16/19 掉到 13/19，Benchmark v2 Silver 从 .690 掉到 .262。
这是本轮最重要的负结果。

---

## 1. 数据来源与许可证

### A. Amazon ESCI

- 本地路径：`data/esci_external/shopping_queries_dataset_examples.parquet`
- 本地路径：`data/esci_external/shopping_queries_dataset_products.parquet`
- 官方仓库：https://github.com/amazon-science/esci-data
- 本地数据 commit：`7916cdf6ab75a462e77f20ab40428a10923998d5`
- 使用：`small_version=1`、`product_locale=us`、官方 `train` / `test`。
- 官方 E/S/C/I 人工相关性标签原样保留。
- 许可证：**Apache License 2.0**（ESCI 仓库 LICENSE）；本报告不重新分发原始数据。
- **没有**使用 BGE 对未评判商品重检索后再转移旧标签；候选只使用官方已评判商品。

### B. ShopRank-Bench

- 本地路径：`data/shoprank_bench/`
- 来源：https://huggingface.co/datasets/srpone/zoowork-shoprank-bench
- 许可证：**CC BY-NC 4.0（non-commercial）**。
- 结构：10,511 preference pairs / 1,500 AHP / 1,302 budget。
- 标签来自跨族 LLM Judge Panel；**不是 HUMAN_GOLD**。
- 与 ShopRanker 发布方同源，不能描述为完全独立于开发方的外部验证；只能作为公开 judge preference benchmark。

### C. PersonaShop

- 本地路径：`data/personashop/questions_test.jsonl`
- 来源：https://huggingface.co/datasets/PersonalizedAgents/PersonaShop
- 许可证：`cc-by-4.0-and-source-terms`（Amazon Reviews 2023 source terms 适用）。
- 使用 `questions` test（327 rows / 2,415 questions / 231 users）与 profile 字段。
- Profile 是 GPT-4o 从历史评论总结的 **inferred preference**，不是 Event-Sourced Memory。
- **没有**把 target rating / target review 暴露给模型。

### D. AgentRec-X

- P20 frozen regression。
- Benchmark v2 300 scenarios。
- Phase 1 16 counterfactual cases。
- 这些结果不与 ESCI、ShopRank-Bench、PersonaShop 混算。

---

## 2. ESCI 可拆分 Query 的真实数量

从 ESCI US `small_version=1` 的官方 Train/Test 中，按可验证属性词典与简单规则提取 **当前 Query 的偏好 clause**。

```text
extractable queries: 639
used with >=2 judged candidates: 630
candidate rows: 11,352
train queries: 431
val queries: 34
test queries: 165

train rows: 7,612
val rows: 636
test rows: 3,104

positive preferences: 10,935 rows
negative preferences: 417 rows
product text coverage: 100%
```

拆分条件：

- 原始 Query 含可识别商品类别词；
- 删除 Preference clause 后 Base Query 仍 >=2 tokens 且含商品类别词；
- clause 确实来自原 Query；
- 记录 original query / base query / preference / polarity / clause / rule。

无法可靠拆分的 Query 不强行制造偏好，保留在普通 relevance 轨道外，不进入本实验。

**查询来源标记：**

```text
preference_source = query_derived_proxy
```

不是 `explicit_historical_memory`。

### Split 与排除

- Train：官方 ESCI Train，按 query_id hash 90/10 分 train/val；同一 query 所有变体保持同一 split。
- Final Test：官方 ESCI Test，**排除 Benchmark v2 使用过的 9 个重叠 query**。
- 商品 ID 可以跨 Query，但 Query Family 不跨 split。
- Benchmark v2 的 100 个 Query 不再充当全项目 untouched holdout；本实验的 final test 是 ESCI Test 中未与 Benchmark v2 重叠的 165 个可拆 Query。

### ESCI 标签

| Label | Meaning | Grade |
|---|---|---:|
| E | Exact | 3 |
| S | Substitute | 2 |
| C | Complement | 1 |
| I | Irrelevant | 0 |

**C 保持 grade 1，不合并进 S。**
主指标按记录的 graded nDCG 计算；不声称与官方任务实现逐位完全一致，映射已明确记录。

数据/特征文件：

- `data/public_preference_ranking_v1/esci_preference_dataset.parquet`
- `data/public_preference_ranking_v1/query_preferences.jsonl`
- `data/public_preference_ranking_v1/dcn_features.parquet`
- `data/public_preference_ranking_v1/dcn_feature_schema.json`
- `data/public_preference_ranking_v1/manifest.json`

---

## 3. Base Query / Preference Dataset 与三组特征

### A. Base-Query Only

```text
base query lexical features + product metadata/attributes
NO preference features
NO full-query lexical features
```

62 features。

### B. Base-Query + Structured Preference

```text
A features
+ preference_present / match / conflict / unknown / coverage
+ preference_is_negative
+ pref_attr_* one-hot
```

111 features。

### C. Original Full Query

```text
full query lexical features + same product metadata/attributes
NO structured preference features
```

62 features。

所有三组使用：

- 同一 candidate rows；
- 同一官方 E/S/C/I 标签；
- 同一 Train/Val/Test；
- 同一 pair 预算 / 同一训练预算；
- 同一 DCN 架构。

### 特征与泄漏控制

- 不把 `query_id` / `product_id` 作为模型输入；
- A 不接收拆出的 Preference，也不接收完整 Query 的 lexical 特征；
- C 接收完整 Query，但不接收结构化 Preference；
- B 接收结构化 Preference，但 base query 不包含被拆出的 clause；
- 商品属性只从商品 title/description/bullet/color 证据检测；
- 商品未提及某属性时是 UNKNOWN，不是确认不具备；
- 没有外部未评商品。

---

## 4. DCN-V2 三组真实训练

模型：`experiments/dcn_v2/model.py` DCN-V2 / DCN-Mix；hidden 64，3 cross layers，dropout 0.1。
损失：同 Query 内 pairwise BCE(winner − loser)。
优化：AdamW lr 5e-4，weight decay 1e-4，grad clip 1.0，max 30 epochs，early stop patience 8，seed 42。

### 训练规模

| | A | B | C |
|---|---:|---:|---:|
| Features | 62 | 111 | 62 |
| Parameters | 33,217 | 39,489 | 33,217 |
| Train groups | 431 | 431 | 431 |
| Val groups | 34 | 34 | 34 |
| Test groups | 165 | 165 | 165 |
| Train pairs | 35,038 | 35,038 | 35,038 |
| Best epoch | 1 | 2 | 1 |
| Train seconds | 19.243 | 22.828 | 19.576 |
| Peak GPU | 19.64 MB | 19.84 MB | 19.84 MB |

### Test 指标

| Arm | nDCG@3 | nDCG@5 | nDCG@10 | MRR | Pairwise | Pairs |
|---|---:|---:|---:|---:|---:|---:|
| A Base Query Only | 0.767278 | 0.790064 | 0.916968 | 0.654863 | 0.582875 | 15,439 |
| B Base + Preference | **0.797396** | **0.806033** | **0.927252** | **0.720003** | 0.584753 | 15,439 |
| C Full Query | 0.780677 | 0.801618 | 0.919585 | 0.671316 | **0.588380** | 15,439 |

### 按 preference polarity

| Arm | Positive nDCG@10 / MRR | Negative nDCG@10 / MRR |
|---|---:|---:|
| A | 0.923194 / 0.668159 | 0.776441 / 0.354762 |
| B | 0.930613 / 0.719201 | **0.851378 / 0.738095** |
| C | 0.925775 / 0.684602 | 0.779864 / 0.371429 |

Negative 只有 **7 个 test groups**，不能把 B 的大幅提升当成稳定结论。

### 按商品 preference evidence coverage

| Arm | coverage<0.5 nDCG@10 / MRR | coverage>=0.5 nDCG@10 / MRR |
|---|---:|---:|
| A | 0.911512 / 0.720062 | 0.918491 / 0.636668 |
| B | **0.918189 / 0.690201** | **0.929781 / 0.728319** |
| C | 0.910342 / 0.692593 | 0.922164 / 0.665378 |

### Level 1/2 解释

- **Level 1**：B 在 ESCI 官方相关性标签上 nDCG@10 高于 A（+0.010284），MRR +0.06514。
- **Level 2**：B 的增益来自“显式结构化当前 Query 偏好”，是 query-derived proxy；不能解释为长期 Memory。
- C 的 pairwise 略高，说明完整 Query 文本仍包含模型可利用的属性信号；但 B 的 nDCG/MRR 更高，结构化偏好对排序位置更有效。

文件：

- `runs/public_preference_ranking_v1/dcn_training_results.json`
- `runs/public_preference_ranking_v1/A_base_query_only.pt`
- `runs/public_preference_ranking_v1/B_preference_on.pt`
- `runs/public_preference_ranking_v1/C_full_query.pt`
- `runs/public_preference_ranking_v1/esci_test_predictions.jsonl`（495 行 = 165 groups × 3 arms）

---

## 5. ShopRank-Bench 公开评测

ShopRanker-0.6B，frozen，official prompt + yes/no logit，structured product text。

| Track | Rows | Accuracy |
|---|---:|---:|
| Preference | 10,511 | **0.765294** |
| AHP | 1,500 | 0.316667 |
| Budget | 1,302 | 0.798003 |

### Preference tiers

| Tier | n | Accuracy |
|---|---:|---:|
| gold | 1,843 | 0.911557 |
| silver | 4,445 | 0.799325 |
| bronze | 4,223 | 0.665641 |

### Qwen3-Reranker-0.6B base

固定前 1,000 个 preference pairs（非调参子集）：

```text
overall: 0.736
gold:    0.865169
silver:  0.785219
bronze:  0.622108
```

### AHP / Budget 诊断

AHP overall 31.67%，明显低于 pairwise 机会水平；按 hierarchy_high/low 也没有稳定遵循属性优先级：

- hierarchy_high=style: 0.02
- hierarchy_high=audience: 0.112
- hierarchy_high=product_type: 0.4775
- hierarchy_high=color: 0.58
- hierarchy_high=price: 0.5533

Budget 79.80%，但没有被拆成“理解预算约束”的证明；它只是公开诊断。

### 限制

- Judge labels 是 LLM panel 标签，不是真人消费者偏好；不得称为 HUMAN_GOLD。
- 与 ShopRanker 同源，不能作为独立开发方验证。
- 只跑了 structured product text；natural 渲染未跑，节省 GPU 成本。
- 结果不与 ESCI nDCG 合并。

文件：`runs/public_preference_ranking_v1/shoprank_bench.json`

---

## 6. PersonaShop 实际可评测能力

使用 `questions` test：

```text
327 target products
2,415 generated questions
231 unique users
```

两种输入：

- A. Question + Target Product（无 Profile）
- B. Question + User Profile + Target Product

商品文本只包含 title/category/store/price/features/description/details；**target rating 和 target review 从未进入模型输入**。

### 结果（不是 Accuracy）

```text
A mean score: 0.635458
B mean score: 0.321838
delta mean:   -0.313620
profile raises: 9.36%
profile lowers: 89.69%
elapsed: 75.5 s
peak GPU: 3,987.88 MB
```

解释：

- PersonaShop 没有对应的 pairwise human preference gold，**不能计算/报告 Accuracy**。
- Profile 被拼接进 Query 后，ShopRanker 对 target product 的**绝对 fit score** 系统性下降；这可能是长 Profile 文本、Query 分布偏移或模型不适应，而不是“个性化正确”。
- 该结果只能说明：**不能把 PersonaShop profile 直接当 Event-Sourced Memory，也不能把 score 下降解释为个性化失败或成功**。
- 需要人工逐例核查，或未来用带人工标准答案的 profile-conditioned pair 才能评估。

文件：`runs/public_preference_ranking_v1/personashop.json`

---

## 7. AgentRec-X Shadow

### 7.1 Feature Adapter

将 ESCI 训练特征映射到 Agent 候选：

- Title → title_len / product tokens；
- Agent canonical attributes → product_attr_*；
- Query 与 title/attrs 的 token overlap → base/full lexical features；
- Active memory 的第一个 preference → preference match / conflict / UNKNOWN；
- 无 description / bullet / brand → 0，**不伪造为有效证据**；
- 缺失属性 → UNKNOWN，不是 negative；
- 使用 ESCI train 的 mean/std，不重新拟合 Agent 标签。

### 7.2 P20

19 recommend tasks；frozen P20；不训练/不调参。

| Arm | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---:|---:|---:|---:|
| DCN A Base Query Only | **16/19** | 18/19 | 18/19 | 0.883772 |
| DCN B Preference ON | 13/19 | 17/19 | 18/19 | 0.788304 |
| DCN C Full Query | 15/19 | 17/19 | 18/19 | 0.852339 |
| Existing Scorer | 18/19 | 19/19 | 19/19 | 0.973684 |
| ShopRanker Query Only | 18/19 | 18/19 | 18/19 | 0.956140 |
| ShopRanker Query+Memory | 19/19 | 19/19 | 19/19 | 1.000000 |

`0051`：

```text
DCN A top1 = B08D32JM5F (acceptable)
DCN B top1 = B07BKPKXTT (acceptable rank 3)
DCN C top1 = B07BKPKXTT (acceptable rank 2)
```

**Preference ON 在 P20 上让 DCN 从 16/19 掉到 13/19**；ESCI preference 特征没有迁移到 Agent Memory。

### 7.3 Benchmark v2 Silver

145 个 SILVER pair：

| Arm | Pairwise | Correct |
|---|---:|---:|
| DCN A | **0.689655** | 100/145 |
| DCN B | 0.262069 | 38/145 |
| DCN C | **0.703448** | 102/145 |

Top-1 变化 vs A：

```text
B: 217 / 300
C: 45 / 300
```

Preference ON 在 Benchmark v2 上严重破坏 Silver 排序；与 ESCI 上的正增益相反。
这明确说明 **Level 2 的偏好代理收益不能直接迁移到 Level 4 的 Event-Sourced Memory 场景**。

### 7.4 Counterfactual Cases

16 个既有 cases：

```text
cases: 16
top1_changed: 12/16
full_changed: 15/16
requires_clarification: 2
```

没有把变化称为“正确”；这只是 DCN 对 Memory 状态的 mechanistic response。Conflict 仍标记 ASK。

### 7.5 Shadow 成本

```text
DCN 20-candidate pool:
mean 87.642 ms, p50 70.518 ms, max 301.085 ms
GPU peak: 9.58 MB
```

Adapter 特征构造包含在延迟中；GPU 模型本身很小，延迟主要来自 Python/特征组装。

所有新模型仅 Shadow；没有修改生产 Scorer/Gate/Planner/P18。

---

## 8. 四层结论

### Level 1 — Public Relevance Gold

DCN B 在 ESCI 官方 E/S/C/I 标签上优于 A：nDCG@10 +1.03pp，MRR +6.51pp。
这是一个真实、独立于 AgentRec-X 的人工相关性结果。

### Level 2 — Preference-Conditioned Proxy

B 优于 A 说明显式结构化当前 Query 偏好可以改善相关性排序；但偏好是 `query_derived_proxy`，不是历史 Memory。

### Level 3 — Public Judge / Profile

ShopRanker 在 ShopRank-Bench preference 76.53%，Budget 79.80%，AHP 31.67%；Panel labels 不是真人 Gold。
PersonaShop 只能做逐例诊断，不能报告 Accuracy。

### Level 4 — Actual Event-Sourced Memory

**未证明。**
DCN B 在 P20 上 13/19；在 Benchmark v2 Silver 上 0.262，明显低于 A/C；`0051` acceptable 从第 1 掉到第 3。
ESCI 的 relevance/preference 增益没有迁移到 Agent Memory；不能宣称获得个性化 Memory 收益。

---

## 9. 限制

1. **HUMAN_GOLD = 0**：ESCI 是人工相关性，但不是长期 Memory 偏好；ShopRank-Bench 是 LLM Judge；PersonaShop profile 是推断。
2. ESCI 可拆偏好 Query 只有 639 个，used 630；test 165；negative test 仅 7 groups。
3. 偏好从当前 Query 拆出，B 是 proxy 实验，不是历史 Memory 因果实验。
4. Product text coverage 100%，但 Agent Adapter 缺 description/bullet/brand；特征分布与 ESCI 不同。
5. DCN 训练 best epoch 1–2，存在明显 early saturation；没有大规模调参。
6. ShopRank-Bench only structured text；natural 未跑；AHP 明显失败。
7. PersonaShop 无 pairwise gold；score shift 不能解释为个性化效果。
8. Agent Shadow 的 Benchmark v2 仍是 synthetic memory + Silver labels；CF 没有独立方向 Gold。
9. 不把 ShopRank-Bench / PersonaShop 与 ESCI nDCG 或 P20 混合。

---

## 10. 下一轮模型选择建议

1. **保留 DCN B 作为 ESCI relevance/preference-conditioned ranker**，但明确它只是 Level 1/2 模型，不是 Agent Memory ranker。
2. **不要将 DCN B 接入生产 Agent**；P20 13/19、Benchmark v2 .262、0051 退化，已经足够否决。
3. **DCN A/C 也不是生产候选**：A 16/19 低于 Existing/Shop；C 15/19。
4. ShopRanker 仍是最强的 zero-shot P20 baseline，但新 Silver 仅 55.17%，且 negative 方向 8/17；不能直接替换 Scorer。
5. 下一轮若要 Level 4 成功，必须补：
   - 真实 Event-Sourced Memory 状态下的成对/选择监督；
   - 真实曝光或人审 pair；
   - Agent 域内 Memory-conditioned 训练，而不是 ESCI 当前 Query 偏好；
   - 可在 ESCI 预训练后做 Agent-specific calibration，但不能用 P20 provisional 当 Gold。
6. 继续把 ShopRank-Bench / PersonaShop 作为公开诊断，不作为 Human Gold。

---

## 11. 交付文件

### 数据

```text
data/public_preference_ranking_v1/
  esci_preference_dataset.parquet
  query_preferences.jsonl
  dcn_features.parquet
  dcn_feature_schema.json
  manifest.json
  dcn_manifest.json

data/shoprank_bench/
  preference.jsonl / ahp.jsonl / budget.jsonl
  README.md / evaluate.py

data/personashop/
  questions_test.jsonl
  user_profiles_test.jsonl
  README.md

data/memory_ranking_benchmark_v2/   # Phase 4, frozen for Agent Shadow
```

### 代码

```text
experiments/public_preference_ranking/build_esci_proxy.py
experiments/public_preference_ranking/build_dcn_features.py
experiments/public_preference_ranking/train_dcn.py
experiments/public_preference_ranking/predict_esci.py
experiments/public_preference_ranking/shoprank_bench_eval.py
experiments/public_preference_ranking/personashop_eval.py
experiments/public_preference_ranking/agent_shadow.py
```

### 结果

```text
runs/public_preference_ranking_v1/
  dcn_training_results.json
  A_base_query_only.pt
  B_preference_on.pt
  C_full_query.pt
  esci_test_predictions.jsonl
  shoprank_bench.json
  personashop.json
  agent_shadow.json
```

### 复现参数

```text
seed: 42
DCN: hidden 64, cross layers 3, dropout 0.1
optimizer: AdamW lr 5e-4, weight decay 1e-4, grad clip 1.0
loss: pairwise BCE
max epochs: 30; early stop patience: 8
ESCI gains: E=3, S=2, C=1, I=0; C not collapsed into S
```

全部只本地 Commit，未 Push；生产 Scorer/Gate/Planner/P18 未修改，BGE 未重建，ShopRanker 未重训。
