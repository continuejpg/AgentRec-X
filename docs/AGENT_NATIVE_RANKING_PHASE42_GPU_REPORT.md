# Agent-Native Ranking Alignment & GPU Hybrid Shadow — Phase 4.2

> 本轮实际完成 AutoDL RTX 4090 D GPU 推理：ShopRanker 受控 Memory（Natural / Structured）、GPU Hybrid、DCN A/B/C Shadow 复核、Unified Memory Adapter、P20 / Benchmark v2 / Counterfactual 统一对照。
> 没有重新训练 ShopRanker / ESCI DCN / Qwen；生产 Scorer / Gate / Planner / P18 未修改。所有 learned ranker 仍为 Shadow。

---

## 0. 结论先行

### Q1. DCN-V2 退化原因

**没有确认的实现 Bug。**

- Feature Schema / 列顺序 / 输入维度 / normalization 全部核对一致；
- B checkpoint input_dim = 111，与 schema 一致；
- 分数方向没有被搬运反（训练/模型输出仍是 higher = better）；
- 但 B 在 Agent Silver 上的 score margin 与标签 **负相关**：
  - B one-pref：38/145 = 0.2621
  - B full-active：38/145 = 0.2621（Benchmark 多数只有一个 Active Preference，修复 first-only 不改变结果）
  - B memory-off：66/145 = 0.4552
  - A base query：100/145 = 0.6897
  - C full query：102/145 = 0.7034
  - 把 B 分数取反的**诊断值**为 107/145 = 0.7379，但**没有作为修正模型使用**。

结论：**Feature Schema Mismatch / Distribution Shift / Supervision Semantic Mismatch**，不是代码错误。
Agent 的 verified/positive match 特征分布与 ESCI Query-derived preference 差异过大，B 模型学到的 ESCI preference 关系在 Agent 上反向了。错误案例中常见：有 positive match 的候选分数低于 UNKNOWN 候选。

### Q2. Natural vs Structured Memory

| Benchmark v2 Silver | Query Only | Natural Memory | Structured Memory |
|---|---:|---:|---:|
| Pairwise | 77/145 = .5310 | 76/145 = .5241 | **81/145 = .5586** |
| P20 Hit@1 | 18/19 | **19/19** | 18/19 |
| Negative 方向 | 6/17 | **8/17** | 7/17 |

Structured 在 Silver 上小幅高 2.76pp，但在 P20、Negative Direction、Update 上并不更好，且 Top-1 变化更大（127/300 vs 92/300）。
**结论：没有可靠的“Structured 更好”证据；Natural 仍是更稳的 P20 表达。**

### Q3. Hybrid 独立贡献

固定 Hybrid：

```text
pool_z(ShopRanker Query Only)
+ 0.5 * positive_match_rate
- 0.75 * negative_conflict_rate
```

| 指标 | Query Only | Shop Structured | Hybrid |
|---|---:|---:|---:|
| Benchmark v2 Silver | .5310 | .5586 | **.7310** |
| Negative direction | 6/17 | 7/17 | **11/17** |
| P20 Hit@1 | 18/19 | 18/19 | 18/19 |
| Top-1 changes vs Query | — | 127/300 | **33/300** |
| 0051 acceptable top1 | yes | yes | yes |

Hybrid 在 Silver、negative direction 和稳定性上更好，但 **Silver 是规则同源标签**，Hybrid 的 Memory 特征又直接来自同一规则；**不能宣称独立个性化收益**。
Hybrid 是当前最值得保留的 fixed baseline，但仍不是可上线的 Learned Ranker。

### Q4. 是否值得训练 Agent-Native Learned Calibration

**目前证据不足。**
已有 145 Silver 与 Hybrid 规则同源；16 CF 是 mechanistic；P20 provisional。
可以做 forward/backward interface，但没有独立 Agent preference supervision 时，不应把 145 Silver 上的提升宣称为真实偏好学习。
Calibration Head smoke：353 params，backward_ok=true，仅作为未来接口。

### Q5. 能否替换生产 Scorer

**不能。**
P20 Existing 18/19；Shop Natural 19/19；Hybrid 18/19；差异只有 1 个 task。
Benchmark v2 是 Silver，CF 无方向 Gold，negative direction 只有 11/17。
所有模型继续 Shadow，不替换生产。

---

## 1. 实际 GPU 执行

### 1.1 环境

- AutoDL RTX 4090 D 24GB。
- ShopRanker：`srpone/zoowork-shopranker-0.6b`，Frozen，official prompt + yes/no logit。
- DCN checkpoints：`runs/public_preference_ranking_v1/{A_base_query_only,B_preference_on,C_full_query}.pt`，只推理、不训练。
- 单次集中 GPU 运行：ShopRanker 300 scenarios × 3 memory views；P20 19 tasks；CF 16 cases；然后 DCN 诊断。

### 1.2 GPU 成本

| 项目 | 实测 |
|---|---:|
| ShopRanker peak GPU | **4,583.37 MB** |
| ShopRanker Query Only | p50 109.978 ms / mean 110.003 ms |
| ShopRanker Natural | p50 111.292 ms / mean 111.811 ms |
| ShopRanker Structured | p50 120.462 ms / mean 121.845 ms |
| Hybrid ranking | p50 110.09 ms / mean 110.191 ms（含 Query scoring + GPU memory adjustment） |
| DCN 诊断 wall | **11.429 s**，peak 9.58 MB |
| Calibration smoke | 386.184 ms，peak 17.37 MB，353 params |

**Latency 口径：** 20 candidates/scenario，batch call，GPU。
Hybrid 数字不包含离线 Memory Adapter 特征预计算；Adapter 特征本轮在 CPU 预生成。不能把这个数字当成含完整 CPU Feature Adapter 的生产延迟。

---

## 2. Unified Event-Sourced Memory Adapter

代码：`experiments/agent_native_ranking/memory_adapter.py`

### 2.1 支持

- Active Positive Preferences；
- Active Negative Preferences；
- 多个同时有效 preferences；
- Preference Update；
- Preference Withdrawal；
- Conflict / ASK；
- UNKNOWN Evidence；
- Query Soft Preferences 与 Historical Memory 分离；
- Removed / Superseded 不进入 Active 特征。

### 2.2 Candidate Features

```text
positive_match_count
negative_conflict_count
positive_unknown_count
negative_unknown_count
active_preference_count
preference_evidence_coverage
positive_match_rate
negative_conflict_rate
query_soft_preference_match
query_soft_preference_conflict
verified_attribute_match
verified_attribute_conflict
attribute_unknown_count
requires_ask
positive_preference_count
negative_preference_count
```

关键语义：

- 多个 Active Preferences 全部计数，不再只取第一个；
- Negative Soft Preference 不升级为 Hard Constraint；
- UNKNOWN 不当作 absence，也不当作 confirmed conflict；
- Withdrawal 后不再产生该偏好贡献；
- ASK conflict 保持原 Gate 语义，不强行生成正常推荐 Gold。

Data：`runs/agent_native_ranking_phase42/memory_adapter_features.jsonl`

```text
rows: 6,896
benchmark_v2: 6,000
p20:            380
counterfactual: 516
```

---

## 3. DCN-V2 Alignment 诊断

Diagnostics：`runs/agent_native_ranking_phase42/dcn_alignment_diagnostics.json`

### 3.1 变体结果

| Variant | Correct | N | Accuracy |
|---|---:|---:|---:|
| B one-pref (old adapter) | 38 | 145 | 0.262069 |
| B full-active | 38 | 145 | 0.262069 |
| B memory-off | 66 | 145 | 0.455172 |
| A base query | 100 | 145 | 0.689655 |
| C full query | 102 | 145 | 0.703448 |
| B inverted score（仅诊断，绝不使用） | 107 | 145 | 0.737931 |

### 3.2 Schema 检查

```text
B feature count:              111
B checkpoint input_dim:       111
order matches schema:         true
normalization mean finite:    true
normalization std finite:     true
```

### 3.3 Feature Distribution Shift

| Feature | ESCI train mean | Agent mean |
|---|---:|---:|
| preference_present | 0.665922 | 0.500000 |
| preference_match | 0.637940 | 0.500000 |
| preference_conflict | 0.027982 | 0.000000 |
| preference_unknown | 0.334078 | 0.500000 |
| preference_coverage | 0.665922 | 0.500000 |
| preference_is_negative | 0.037966 | 0.000000 |
| base_overlap_rate | 0.625871 | 0.546544 |
| title_len | 137.13 | 109.18 |

Agent 没有 negative Silver pairs（negative 是 UNJUDGED），所以 `preference_conflict` 在 Agent 诊断集为 0；ESCI 有 2.8%。
Agent 的 match/unknown 是 50/50 对称，而 ESCI preference present 66.6%，B 模型在 Agent 分布上出现了 **反向 score response**。

### 3.4 错误分组

```text
positive_memory:      72 errors
unknown_evidence:     13 errors
preference_update:    13 errors
query_memory_competition: 9 errors
```

典型错误：positive match 候选得到更低分，UNKNOWN 候选得到更高分。
例如 `esci_43953`：A 有 positive match，A score 4.546，B score 4.794，模型选 B。
这不是单个 case 的偶发错误，而是分布级反向。

### 3.5 结论

**Confirmed Implementation Bug: 0**
**Feature Schema Mismatch: no dimension/order/normalization mismatch**
**Feature Distribution Shift: yes**
**Supervision / Semantic Mismatch: yes** — ESCI query-derived preference 与 Agent historical memory 的 evidence 状态、negative 覆盖、unknown 分布不同。

Full-active Adapter 没有修复，因为 Benchmark v2 Silver 多数 scenario 只含一个 Active Preference；真正差异是分布/语义，不是 first-only 取数。

---

## 4. ShopRanker Controlled GPU Memory

代码：`experiments/agent_native_ranking/shopranker_controlled_gpu.py`
结果：`runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json`

### 4.1 三种 memory 表达

**Query Only**：原始 Query。

**Natural Memory**（既有格式）：

```text
{query} User preferences: prefer microphone; avoid over_ear.
```

**Structured Memory**（本轮新增）：

```text
Current Shopping Request:
{query}

Active Long-Term Preferences:
Positive:
- microphone: preferred
Negative:
- over_ear: avoided
```

两者使用同一 candidate 文本、同一 product serialization、同一 active memory snapshot；只改变 Query 表达，不改评分头。

### 4.2 Benchmark v2 结果

| Arm | Pairwise | Top-1 changed vs Query | Top-3 set changed | Full changed |
|---|---:|---:|---:|---:|
| Query Only | 77/145 = .5310 | — | — | — |
| Natural Memory | 76/145 = .5241 | 92/300 | 144/300 | 183/300 |
| Structured Memory | 81/145 = .5586 | 127/300 | 202/300 | 300/300 |
| Hybrid | 106/145 = .7310 | 33/300 | 68/300 | 172/300 |

**Natural vs Structured：** Structured 高 2.76pp，但 Top-1 变化更大、全排序 300/300 变化，P20 更差，negative 更差；不能认为 Structured 更可靠。

### 4.3 P20

| Arm | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---:|---:|---:|---:|
| Query Only | 18/19 | 18/19 | 18/19 | 0.956140 |
| Natural Memory | **19/19** | 19/19 | 19/19 | 1.000000 |
| Structured Memory | 18/19 | 19/19 | 19/19 | 0.973684 |
| Hybrid | 18/19 | 19/19 | 19/19 | 0.973684 |
| Existing Scorer | 18/19 | 19/19 | 19/19 | 0.973684 |

**0051**（所有 Shop 版本都保留 acceptable top1）：

```text
B08D32JM5F > B004K6R43G > B07BKPKXTT
```

Query / Natural / Structured / Hybrid 的 0051 都保持这个排名。
没有为了让 Memory 改变 0051 而调 Prompt。

### 4.4 Directionality

**Positive / Update / Unknown / Competition（Silver pairs）**

| Scenario | Query | Natural | Structured | Hybrid |
|---|---:|---:|---:|---:|
| positive_memory (97) | .5464 | .5361 | **.5670** | .7320 |
| unknown_evidence (16) | .4375 | .5000 | .5000 | .7500 |
| preference_update (17) | .5882 | **.6471** | .5294 | .7059 |
| query_memory_competition (15) | .4667 | .5333 | .6000 | .7333 |

**Negative Direction（mechanistic safe-vs-conflict，17 cases）**

| Arm | safe above conflict |
|---|---:|
| Query Only | 6/17 |
| Natural | 8/17 |
| Structured | 7/17 |
| Hybrid | **11/17** |

Structured 并没有修复 negative；Hybrid 也只到 11/17，不能称为可靠负向偏好理解。

### 4.5 Counterfactual

16 cases：

```text
Query Only:     top1 changed 0/16; full changed 0/16  (by construction memory ignored)
Structured:     top1 changed 4/16; full changed 15/16
Requires ASK:   2
```

By category：

- positive: structured top1 0/4; full 3/4
- negative: top1 1/2; full 2/2
- update: top1 0/2; full 2/2
- withdrawal: top1 1/2; full 2/2
- conflict: top1 1/2; full 2/2（ASK，不能当正常 Gold）
- unknown: top1 1/4; full 4/4

**Withdrawal：** Structured Memory 在 mutated state 中没有保留被撤回偏好；排序变化说明输入状态改变，但没有独立方向 Gold，不能证明“正确撤回”。
**UNKNOWN：** 结构中没有把 UNKNOWN 写成 absence；但模型仍可能把无证据候选排前/后，不能宣称一致。

---

## 5. GPU Hybrid

### 5.1 固定公式

```text
semantic_z = pool_z(ShopRanker Query Only)
memory_adjust =
    +0.5 * positive_match_rate
    -0.75 * negative_conflict_rate
hybrid = semantic_z + memory_adjust
```

- 权重在查看本轮 benchmark 结果前固定；
- 没有调参；
- 没有根据 P20/Silver 搜索；
- 不使用 Query Soft 作为历史 Memory；
- Hard Constraint 仍由 Eligibility / Gate 负责。

### 5.2 实际 GPU 执行

`shopranker_controlled_gpu.py` 在同一个 GPU 运行中：

1. 加载 ShopRanker；
2. 批量计算 Query Only / Natural / Structured scores；
3. 在 GPU tensor 上计算 pool z-score 与 memory adjustment；
4. 返回 eligible rankings；
5. 保存每个候选的 semantic/natural/structured/hybrid 分数。

`hybrid_gpu_predictions.jsonl`：6,000 rows（300 scenarios × 20 candidates）。

### 5.3 结果

| Metric | Value |
|---|---:|
| Benchmark v2 Silver | **106/145 = .7310** |
| Negative direction | **11/17** |
| P20 Hit@1 | 18/19 |
| Top-1 changed vs Query | 33/300 |
| 0051 | acceptable top1 |
| GPU peak | 4,583.37 MB |

Hybrid 在 Silver 和 negative direction 上优于 Shop 单独版本，同时比 Structured 少扰动（top1 33 vs 127）。
但 Silver 是规则同源，Hybrid 的 memory 项又直接使用同一规则特征；**这不是独立个性化证据**。
它是目前最可取的固定 baseline，但只能 Shadow。

---

## 6. 统一 Shadow 对照

所有结果分开报告，不合成总分。

### 6.1 P20（provisional）

| Ranker | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Existing Scorer | 18/19 | 19/19 | 19/19 | 0.973684 |
| ShopRanker Query Only | 18/19 | 18/19 | 18/19 | 0.956140 |
| ShopRanker Natural Memory | **19/19** | 19/19 | 19/19 | 1.000000 |
| ShopRanker Structured Memory | 18/19 | 19/19 | 19/19 | 0.973684 |
| Hybrid | 18/19 | 19/19 | 19/19 | 0.973684 |
| DCN A Base Query | 16/19 | 18/19 | 18/19 | 0.883772 |
| DCN B Preference ON | 13/19 | 17/19 | 18/19 | 0.788304 |
| DCN C Full Query | 15/19 | 17/19 | 18/19 | 0.852339 |

### 6.2 Benchmark v2（Silver 145）

| Ranker | Pairwise | Correct |
|---|---:|---:|
| Existing Scorer | 1.000* | 145/145 |
| Shop Query Only | 0.531034 | 77/145 |
| Shop Natural Memory | 0.524138 | 76/145 |
| Shop Structured Memory | 0.558621 | 81/145 |
| Shop + Memory Hybrid | **0.731034** | 106/145 |
| DCN A | 0.689655 | 100/145 |
| DCN B | 0.262069 | 38/145 |
| DCN C | 0.703448 | 102/145 |

\* Existing Scorer 与 Silver 规则同源，1.0 是 tautological，不是独立质量。

### 6.3 Counterfactual（mechanistic，无方向 Gold）

| Arm | top1 changed | full changed |
|---|---:|---:|
| Query Only | 0/16 | 0/16 |
| Structured | 4/16 | 15/16 |
| DCN B | 12/16 | 15/16 |
| ASK conflicts | 2 | 不生成正常 Gold |

---

## 7. 重点案例

### 7.1 0051

商品：`B08D32JM5F`（acceptable）、`B07BKPKXTT`、`B004K6R43G`。

| Arm | Top-1 | 备注 |
|---|---|---|
| BGE Query Relevance | 见候选池 | 不做额外声明 |
| Shop Query Only | B08D32JM5F | 保持 acceptable |
| Shop Natural | B08D32JM5F | 保持 |
| Shop Structured | B08D32JM5F | 保持 |
| Hybrid | B08D32JM5F | 保持 |
| DCN A | B08D32JM5F | 保持 |
| DCN B | B07BKPKXTT | acceptable 降到第 3 |
| DCN C | B07BKPKXTT | acceptable 第 2 |

DCN B 在 0051 上失败；Shop/Hybrid 没有为了 Memory 把 acceptable 挤出 top1。

### 7.2 Negative Preference

Negative direction 只有 6-11/17；Structured 7/17、Hybrid 11/17。
这不是“负向偏好已被正确理解”，只是固定 Memory 调整改善了部分有明确证据的冲突；对 UNKNOWN/soft negative 仍不可靠。

### 7.3 Withdrawal

Structured 在 withdrawal 后不再使用被撤回偏好（mutated memory 为空）；但排序变化方向没有独立 Gold，不能宣称正确。
Query Only 对 withdrawal 不变（memory ignored），是安全但无个性化的对照。

### 7.4 UNKNOWN

Structured/Hybrid 都保留 UNKNOWN 语义，没有写成 absence。
但 UNKNOWN 候选并不保证排在 verified 候选之后；方向指标 unknown_evidence 也只有 .5-.75，不能宣称一致推理。

---

## 8. 结论与下一步

### Q1
DCN B 退化不是实现 Bug，而是 **Feature Distribution / Supervision Semantic Mismatch**；Feature Schema 和 normalization 已核对。B 在 Agent Silver 上出现反向 score response；full-active 不能修复，因为主要问题是语义/分布，不是 first-pref-only。

### Q2
Natural vs Structured：Structured 在 Silver 小幅高（.5586 vs .5241），但 P20 更差、negative/update 更差、排序更不稳定。**没有可靠证据证明 Structured 更好。**

### Q3
Hybrid 在 Silver（.731）和 negative direction（11/17）上最好，同时 Top-1 扰动小（33/300），0051 保持 acceptable。
但它是固定规则融合，且 Silver 同源，不能称为 Learned Ranker；只能作为 Shadow baseline。

### Q4
**暂不建议训练 Agent-Native Learned Calibration Head。**
缺少独立 Agent preference supervision；145 Silver 与规则同源；16 CF 无方向 Gold；P20 只有 provisional。
Calibration Head 仅完成 353-param forward/backward smoke（backward_ok=true），不作正式训练。

### Q5
**不替换生产 Scorer。**
P20 Existing 18/19，Shop Natural 19/19，Hybrid 18/19；差异只有 1 个 task；Silver 和 CF 都不足以证明真实用户偏好。继续 Shadow。

### 下一阶段建议

1. 保留 **Shop Natural** 与 **fixed Hybrid** 作为两个 Shadow baseline。
2. 为 Hybrid 增加独立 Agent Memory 成对标签；至少数百到数千条。
3. 在独立标签上比较 Natural/Structured/Hybrid，不调 Prompt 权重。
4. 只有独立验证显著且方向一致性提升后，才训练小型 Learned Calibration Head。
5. 不重新训练 ESCI DCN B；不把 ESCI relevance 收益迁移到 Agent Memory。

---

## 9. 交付产物

### 代码

```text
experiments/agent_native_ranking/memory_adapter.py
experiments/agent_native_ranking/build_adapter_features.py
experiments/agent_native_ranking/shopranker_controlled_gpu.py
experiments/agent_native_ranking/dcn_diagnostics.py
experiments/agent_native_ranking/shadow_comparison.py
experiments/agent_native_ranking/calibration_head.py
experiments/agent_native_ranking/build_counterfactual_diagnostics.py
experiments/agent_native_ranking/build_experiment_manifest.py
```

### 结果

```text
runs/agent_native_ranking_phase42/
  dcn_alignment_diagnostics.json
  memory_adapter_features.jsonl
  shopranker_controlled_gpu.json
  hybrid_gpu_predictions.jsonl
  shadow_comparison.json
  counterfactual_diagnostics.json
  gpu_runtime.json
  calibration_smoke.json
  experiment_manifest.json
```

### 约束

```text
ShopRanker fine-tuned: no
ESCI DCN retrained: no
Qwen Planner trained: no
BGE rebuilt: no
Production Scorer/Gate/Planner/P18 modified: no
HUMAN_GOLD added: 0
Git Push: no
```

**Phase 4.2 完成：完成 GPU 受控 Memory 推理、DCN 迁移诊断、统一 Memory Adapter、固定 Hybrid GPU Shadow；结论是 DCN B 为分布/监督语义不匹配，Natural 未被 Structured 证明更优，Hybrid 是当前最有用的固定 Shadow baseline，但尚不足以训练或替换生产。**
