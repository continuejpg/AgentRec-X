# AgentRec-X — Phase 5.2: Retrieval Coverage, Evidence Quality & Final Evaluation

> 本轮不训练模型、不扩大 P100、不引入新 Ranker。目标是定位 P100 推荐质量的实际瓶颈，修复/明确已确认的系统问题，并形成可用于 Demo/面试的最终评测结论。
> 生产默认仍为 `PreferenceAwareScorer`；HUMAN Memory Preference Gold = 0。

---

## 0. 最终结论

1. **最大瓶颈不是 Ranker，而是 Catalog / Judgment / Retrieval 覆盖。**
   - 1,971 个 ESCI Judged Products 中，只有 61 个能精确匹配当前 Electronics Catalog；
   - 只有 60 个在 Accepted BGE Index；
   - 52 个是 E/S 正例；
   - 只有 14 个 E/S 正例进入 BGE Top-20；
   - Eligibility 没有再损失；
   - 最终 Top-K 只有 4 个 E/S 正例。
   - 条件 Recall@20 只有 **14/52 = 26.9%**，且它只是在“已评判且已入 Index 的正例”上的条件召回。

2. **BGE 在公开 Judged Track 上明显强于 ShopRanker Query Only。**
   - Original BGE +0.1106 nDCG@10（95% paired CI `[0.0778, 0.1445]`）；
   - BGE 在 99 个有效 Query 中 76 个 nDCG@10 更高；
   - ShopRanker Query Only 的 MRR@E/S 更高（0.9438 vs 0.9132），但整体排序质量更低；
   - 因此 **ShopRanker Natural 没有资格作为生产默认替换**。

3. **Hybrid 没有显示真实 Memory 收益。**
   - Positive Top1 match 61.7%，低于 Natural 93.6% 和 Existing 89.4%；
   - Negative conflict ON 41.7%，高于 Existing 16.7%；
   - Top1 OFF/ON 仅 11%，弱于 Natural 35% / Existing 34%。

4. **没有证据支持升级生产 Ranker。**
   Existing Scorer 保留生产默认；ShopRanker Natural 保留 Shadow/语义排序候选；Hybrid 保留 Shadow；DCN-V2 仅为历史训练资产。

---

## 1. Retrieval / Judgment Coverage Waterfall

结果：`runs/agentrec_x_phase52/p100_retrieval_judgment_waterfall.json`
逐 Query：`p100_retrieval_coverage_per_query.csv`

### 1.1 全局漏斗

| Stage | Count | 说明 |
|---|---:|---|
| ESCI Judged Products | 1,971 | 官方人工评判商品 |
| E/S positive | 1,597 | 已知相关正例 |
| 精确匹配 Catalog | 61 | Catalog 商品 ID 交集 |
| Accepted BGE Index | 60 | 实际可检索 Index |
| BGE Top-20 | **14** | Original BGE 检索命中 |
| 通过 Eligibility | 14 | 没有额外损失 |
| 进入最终 Top-K | **4** | 三臂 union |

### 1.2 条件 Recall

```text
Indexed E/S positive:                 52
Retrieved in BGE Top-20:              14
Conditional Recall@20:             26.9%
```

这是 **conditional recall on judged + indexed positives**，不是整个商品空间的真实 Recall；未评判商品没有作为负例。

### 1.3 49 个 Catalog overlap Query

- 49 个 Query 至少有一个 judged product 精确匹配 Catalog；
- 44 个 Query 有 Catalog 内 E/S 正例；
- 只有 **13 个 Query** 的 BGE Top-20 含 E/S 正例；
- 最终只有 **4 个正例**进入 Top-K。

### 1.4 13 个 Query 覆盖低的原因

不是“BGE 只覆盖这 13 个 Query”，而是：

1. ESCI US 的 Product ID 与当前 Electronics Catalog 的 accepted IDs 几乎不重合：1,971 → 61；
2. 这 61 个 Catalog 商品只有 60 个进入 Accepted BGE Index；
3. 即使 Query 有 Catalog overlap，其正例也只有 14/52 被 Top-20 检索；
4. Eligibility 不构成主损失；
5. Top-K 再损失 14→4。

**主损失：Catalog/Index 覆盖率；次损失：Top-20 Retrieval；Eligibility 不是主要瓶颈。**

---

## 2. Evidence Quality

结果：`runs/agentrec_x_phase52/p100_evidence_unsupported_cases.jsonl`

### 2.1 失败数量

| Arm | Unsupported / Valid Top-K |
|---|---:|
| Existing | 8 / 258 |
| Natural | 10 / 247 |
| Hybrid | 15 / 252 |
| **Total** | **33** |

### 2.2 原因分类

33/33 全部为：

```text
no_verified_attributes_or_evidence
```

即：

- Product metadata 存在；
- 但该商品的 Verified Attributes 和 Evidence 都为空；
- Top-K Presenter 的 `evidence_supported` 判定为 false；
- 没有出现：
  - Metadata Adapter 属性映射错误；
  - Presenter 使用非 Verified 属性；
  - Evidence Validator 口径不一致；
  - Product ID 不一致。

### 2.3 结论

**这是数据覆盖问题，不是实现 Bug。**
没有放宽验证标准，也没有伪造属性。证据不足的商品保留在 Top-K 中但不冒充“已支持”。

---

## 3. Memory Writer 异常

结果：`runs/agentrec_x_phase52/p100_memory_writer_anomalies.json`

P100 100 个 ON task 中 Existing 状态一致性 **91/100**；9 个异常分为两类：

### 3.1 7 个 unknown_evidence

原始 synthetic event：

```text
I prefer a verified unobtainium feature.
```

Writer 实际输出：

```text
value = "verified unobtainium feature", polarity = prefer
```

但 Executor canonicalisation 只接受 Catalogue 可映射的属性，因此 prepared active preferences 为空。

分类：

```text
writer_produced_unknown_preference_but_executor_canonicalisation_dropped_it
```

处理：

- 标记为 **UNSUPPORTED_SCENARIO**；
- 不把该 case 计为 Verified Match 或 Confirmed Conflict；
- 不绕过 Writer 直接注入 Active Memory；
- 不修改生产 Memory Writer / Scorer。

### 3.2 2 个 multiple_preferences

| Query | 原始表达 | Writer 输出 |
|---|---|---|
| p100_034 | I prefer {attr} and {second}. | 只提取 1 个 |
| p100_092 | I prefer {attr} and {second}. | 只提取 1 个 |

分类：

```text
extractor_only_one_preference
```

其他 9 个 multiple_preferences 均正确提取 2 个。
这是 Rule Extractor 对“单句 and 连接”的覆盖限制；标记为 **partial_extraction**，未为此修改生产 Extractor。

### 3.3 结论

- 不是 Memory State 丢失；
- 不是 Reader 返回错误；
- 7 个 unknown 是“无法映射到 Catalogue verified attribute 的文本”，按 unsupported 处理；
- 2 个 multiple 是测试表达导致的 partial extraction；
- 没有直接注入或改写 Active Memory 来掩盖问题。

---

## 4. ASK / No-Verified / Planner Error

结果：`runs/agentrec_x_phase52/p100_ask_analysis.json`

300 条 E2E 状态：

```text
finalized:                         154
awaiting_clarification:            140
error (planner):                     3
no_verified_candidate_in_scope:      3
```

### 4.1 ASK 来源

```text
insufficient_decision_criteria:              117
candidate_conflicts_with_stated_preference:   17
planner-direct ASK with no Gate reason:        6
```

按 Memory 场景：

| Scenario | Existing ASK | Natural ASK | Hybrid ASK |
|---|---:|---:|---:|
| no_memory | 25 | 25 | 25 |
| preference_withdrawal | 9 | 9 | 9 |
| negative_memory | 4 | 7 | 6 |
| unknown_evidence | 6 | 6 | 6 |
| preference_update | 1 | 1 | 1 |
| positive / multiple / competition | 0 | 0 | 0 |

主要来自：

1. 用户需求本身缺少必要条件；
2. 商品属性 UNKNOWN / 无 Evidence；
3. 部分 negative Memory 使 Gate 进入冲突/证据不足；
4. 17 个 conflict 是合理 ASK。

### 4.2 p100_008

`p100_008` 三个 arm 都是 Qwen invalid action：

```text
HttpPlannerError: unknown planner action ''
```

记录为 error；没有跳过 Validator，没有强制 FINALIZE，没有自动改 ASK。

### 4.3 p100_063

`motorcycle helmet earphone` 三个 arm 都是：

```text
no_verified_candidate_in_scope
```

只有 20 个 eligible candidates，但没有满足 updated memory 的 verified candidate。
不是推荐成功，也不是 FINALIZE。

### 4.4 结论

**ASK 不是失败的自动同义词。**
No-Memory / Withdrawal 的 ASK 主要是“缺少必要条件/证据”，不是 Ranker 排序错误；没有为了 FINALIZE Rate 修改 Gate 阈值。

---

## 5. 公开人工标签下的排序质量

### 5.1 P100 Judged Track

Candidate Pool：**官方 ESCI Judged Products**，不是 Original BGE Top-20。

| Model | nDCG@3 | nDCG@5 | nDCG@10 | MRR(E/S) | Pairwise Acc |
|---|---:|---:|---:|---:|---:|
| Original BGE Query Baseline | **0.920661** | **0.890269** | **0.849054** | 0.913171 | **0.677507** |
| ShopRanker Query Only | 0.693587 | 0.705226 | 0.739697 | **0.943845** | 0.654720 |
| Existing Scorer | not comparable | — | — | — | — |
| Natural Memory | no memory; Query Only is the non-personalized comparison | — | — | — | — |
| Hybrid | not defined | — | — | — | — |

Paired BGE − ShopRanker nDCG@10：

```text
mean difference: +0.110584
95% bootstrap CI: [0.077754, 0.144511]
BGE wins: 76 / 99 comparable queries
ShopRanker wins: 22
ties: 1
```

**BGE 在公开相关性排序上明显更强。**
ShopRanker 的 MRR 更高说明它更常把 E/S 放在第 1 位；但 nDCG@3/5/10 和 Pairwise 更差，说明多等级排序整体更弱。

### 5.2 解释边界

- 这证明的是 **公开 Query-Product Relevance**；
- 不证明长期用户 Memory 收益；
- Existing Scorer 不具 Catalog 属性 schema，不能强行比较；
- 没有为了凑三模型对照而虚构属性。

---

## 6. 独立偏好评测

复用 Phase 4.4 的 40 对 Blind Review Tool。

当前：

```text
HUMAN_SINGLE_REVIEW = 0
HUMAN Memory Gold = 0
```

若没有人工 CSV：

- 不计算模型 Pairwise Agreement；
- 不伪造标签；
- 不阻塞其他任务。

没有使用 LLM 作为人工 Gold；若未来使用，只能标记 `LLM_JUDGE_SILVER`，不能与 HUMAN/ESCI Gold 混合。

---

## 7. GPU / 延迟口径

必须区分：

### 7.1 ShopRanker 实际 GPU Scoring

```text
Phase 5.1 P100 ranker GPU:
  wall: 34.16s
  Agent rows: 2,000
  Memory modes: OFF/ON
  Judged products: 1,971
  Peak allocated: 6,611 MB
```

这是真实模型前向 + tokenization 的批量成本，不是 0.6ms。

### 7.2 E2E Ranker Injection

```text
Phase 5.1 E2E Natural/Hybrid p50: 0.60–0.61 ms
```

这是 **已计算分数重新排序 / 注入** 的延迟，不包含 ShopRanker 模型前向。
不能把它称为 ShopRanker 实时推理延迟。

### 7.3 其他

```text
Memory Adapter CPU:      0.0704 ms / 20 candidates
Qwen Planner p50:        3.26–3.35 s
Full Agent E2E p50:      10.80–11.49 s
BGE Judged baseline:     2.07 s / 1,971 products; peak 514 MB
OOM:                     0
```

---

## 8. 最终模型决策

### 生产默认

**Existing PreferenceAwareScorer。**

原因：

- P100 公开 Judged Track 上 BGE 直接排序明显强于 ShopRanker Query Only（nDCG@10 0.8491 vs 0.7397）；
- ShopRanker Natural 没有独立 Memory Gold；
- Hybrid 没有显示稳定 Memory 改善；
- Evidence 缺失和 ASK 主要是数据/覆盖问题，不是 Ranker 更换能解决。

### 最强语义排序候选

**ShopRanker Natural Memory：保留 Shadow。**

- Positive Top1 match 93.6%，高于 Existing 89.4%；
- 但 Negative conflict 50% 高于 Existing 16.7%；
- P100 partial judged hit@5 低于 Existing；
- 无 Human Memory Gold；
- 只能进入受控人工评测，不能替换生产。

### Fixed Hybrid

**保留实验 Shadow。**

- Positive Top1 61.7%；
- Negative conflict 41.7%；
- Top1 OFF/ON 仅 11%；
- 没有真实 Memory 收益证据。

### DCN-V2

**保留训练研究资产，不作为 Agent 生产候选。**

### 差异归因

| 观察差异 | 主要归因 |
|---|---|
| Agent 候选 99.3% UNJUDGED | Catalog / ESCI identity coverage |
| 13 个 Query 有 Top-20 正例 | Catalog/Index + BGE retrieval |
| Top-K 仅 4 个正例 | Retrieval 后 Top-K 呈现 |
| Evidence 33 个 unsupported | 商品 Metadata 无 verified attributes/evidence |
| ASK 45–48% | 缺少必要条件 / UNKNOWN evidence / 合理 conflict |
| 7 unknown Memory 空态 | Writer 输出 unknown 被 executor canonicalisation 丢弃 |
| 2 multiple 少一个偏好 | Rule Extractor 单句 and 覆盖限制 |
| 3 planner error | Qwen invalid action |
| 3 no-verified | 检索/证据不足以支持推荐 |
| Natural positive vs negative 矛盾 | 排序行为，非 Memory 质量 Gold |

---

## 8.5 Q1–Q5 结论

### Q1. ShopRanker 是否仍然优于 Existing？

**没有证据。**

- ShopRanker Natural 的 Positive Top1 match 更高（93.6% vs 89.4%），但 Negative conflict 更差（50.0% vs 16.7%）；
- 13 个精确 ESCI overlap 的 partial Hit@5：Natural 6/13 vs Existing 9/13；
- 在公开 Judged Track 上，Original BGE nDCG@10 0.8491 vs ShopRanker Query Only 0.7397；BGE paired +0.1106 `[0.0778, 0.1445]`；
- Existing 与 Judged Track 不可比较，不能把 BGE 与 ShopRanker 的差距直接说成 Existing 与 BGE 的差距；
- 没有任何独立 Memory Gold。

### Q2. Hybrid 是否真正改善 Memory？

**没有。**

- Positive Top1 match 61.7%，低于 Natural 93.6% 和 Existing 89.4%；
- Negative conflict ON 41.7%，高于 Existing 16.7%；
- Top1 OFF/ON 仅 11%；
- Silver 指标是规则同源，不是真实用户偏好。

### Q3. 100 个任务是否足以做生产决策？

- 100 个有效公开 judged groups，可做公开相关性比较；
- Agent pool 只有 13 个 Query 含精确 ESCI 评判，1,986/2,000 候选 UNJUDGED；
- 真实历史 Memory Preference Gold = 0；
- 因此足够支持 **系统稳定性/公开相关性/覆盖瓶颈** 结论，**不足够支持长期个性化质量或生产替换**。

### Q4. 实际 Agent E2E 是否稳定？

- 300/300 完成；
- 154 FINALIZE / 140 ASK / 3 planner error / 3 no-verified；
- 0 budget_exhausted、0 hard violation、0 OOM；
- E2E p50 10.80–11.49s，Planner p50 3.26–3.35s；
- ASK 主要来自证据不足/合理冲突，不是 Ranker 全面崩溃。

### Q5. 下一步是否还需要训练模型？

**不需要新增训练。**
没有证据支持 ShopRanker/Hybrid 替换 Existing。优先修复 Catalog/Index/Judgment 覆盖、补人工 Memory Preference 评测；不要用 P20 19/19 作为上线理由。

---

## 9. 演示 / 面试结论

可以如实表述：

- 建立了 **P100 独立评测集**；
- 有 **公开 ESCI 人工标签**的 Judged Track；
- 有 **300 条真实 Qwen Agent E2E**；
- 有 **Retrieval/Judgment Coverage 漏斗**，定位到 Catalog/Index 是主瓶颈；
- 有 **BGE 与 ShopRanker 的公开相关性对照**，发现 BGE 更强；
- 有 **Evidence/ASK/Memory 异常定位**，没有用伪造或放宽标准掩盖；
- 生产默认仍为 Existing Scorer；
- 下一步不是再训练 Ranker，而是修复 Catalog/Index 覆盖和做人工偏好评测。

---

## 10. 工程交付

```text
docs/AGENTREC_X_PHASE52_QUALITY_CLOSURE.md
runs/agentrec_x_phase52/
  p100_retrieval_judgment_waterfall.json
  p100_retrieval_coverage_per_query.csv
  p100_evidence_unsupported_cases.jsonl
  p100_memory_writer_anomalies.json
  p100_ask_analysis.json
  p100_judged_bge_predictions.jsonl
  p100_judged_bge_metrics.json
  p100_judged_bge_vs_shopranker.json
  p100_judged_bge_runtime.json
  phase52_summary.json
  phase52_experiment_manifest.json
```

约束：

```text
Qwen / ShopRanker / DCN retrained: no
BGE rebuilt: no
生产默认 Scorer / Gate / Planner / P18 modified: no
Hybrid weights changed: no
P100 expanded: no
Unjudged treated as negative: no
HUMAN_GOLD fabricated: no
LLM judge used as human Gold: no
Git Push: no
```
