# Production Candidate Evaluation & End-to-End Agent Ranking — Phase 4.3

> 本轮完成真实 Qwen Planner + Memory Writer/Reader + Ranker + DecisionGate 的 Agent E2E Shadow，并在 AutoDL RTX 4090 D 24GB 上完成模型生命周期、并发驻留和延迟实测。
> 生产默认仍为 `PreferenceAwareScorer`；没有替换生产 Scorer，没有修改 Gate / Planner / P18，没有重新训练 ShopRanker / Qwen / DCN。
> 所有 P20 结果为 provisional；Benchmark v2 为 Silver；Counterfactual 为 mechanistic；**HUMAN_GOLD = 0**。

---

## 0. 结论先行

### P20 Agent E2E（19 个 recommend tasks）

| Arm | Success | FINALIZE | ASK | Hard Violations | Answer Evidence Supported | E2E p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Existing Scorer | **18/19** | 19/19 | 0 | 0 | 19/19 | 10.87s / 13.43s |
| ShopRanker Natural Memory | **19/19** | 19/19 | 0 | 0 | 19/19 | 10.46s / 13.78s |
| Fixed Hybrid | **18/19** | 19/19 | 0 | 0 | 19/19 | 10.59s / 13.88s |

- `0051`：Existing 选 `B07BKPKXTT`（错误）；Natural 和 Hybrid 均选 acceptable `B08D32JM5F`。
- Natural 在 E2E 中唯一达到 19/19；但只比 Existing 多 1 个 task，**不足以替换生产**。
- Hybrid 的 E2E 0054 失败（选择 `B08PFW1S1V`），没有达到 Natural 的行为一致性。

### 判定

- **Existing Scorer：保留生产默认。**
- **ShopRanker Natural：值得继续 Shadow / 进入下一轮用户偏好评测；暂不进入生产试用。**
- **Fixed Hybrid：保留为实验 Shadow；CF E2E 16/16 budget_exhausted，不能进入试用。**
- **DCN-V2：历史结果引用，不参与本轮生产候选。**

---

## 1. 三个方案的统一实验接口

新增实验目录：`experiments/agent_ranker_e2e/`

统一输入：

- Current Query
- Eligible Candidate Pool
- Candidate Evidence
- Active Memory Snapshot
- Query Soft Preferences
- Arm: `existing | natural | hybrid`

统一输出：

- Ranked Candidate IDs
- Raw Scores / Semantic Scores
- Memory Feature Contributions
- Final Hybrid Scores
- Evidence Coverage
- Latency Breakdown

实现方式：

- 复用生产 `PreferenceAwareScorer`、`ElectronicsPlannerExecutor`、`DecisionGate`、`ShoppingCompanionMemory`、`HistoryStateReader`；
- **没有修改生产 Scorer 本体**；
- 通过实验型 `CachedArmRanker` 注入 ShopRanker/Hybrid 排序；
- ShopRanker 分数来自 Phase 4.2 的真实 GPU 运行；
- 所有 arm 使用同 Query、同 BGE 检索、同 Memory、同 Gate。

---

## 2. 复现差异：Phase 4 vs 4.2

报告文件：`runs/agent_ranker_phase43/reproduction_differences.json`

### 2.1 差异

| 指标 | Phase 4 | Phase 4.2 |
|---|---:|---:|
| Shop Query Only | 80/145 | 77/145 |
| Natural Memory | 80/145 | 76/145 |

145 个 Silver pairs 中 **13 个 pair 的预测发生改变**。

### 2.2 明确原因

```text
290 / 290 参与 pair 的商品文本发生变化
旧序列化: title | category | verified_attributes | evidence | unknown_attributes
         max_length = 4096
新序列化: title | product_type | verified_attributes | evidence | unknown_attributes
         max_length = 1536
```

- `category` → `product_type`；
- evidence 拼接格式从 list 前缀改为 attribute=evidence；
- max_length 4096 → 1536；
- 精确的 13 个 changed pair 已逐条保存；
- 没有更改模型 revision、tokenizer、bf16 precision、yes/no scoring formula；
- 没有用这些差异回头调 Prompt 或选阈值。

### 2.3 Structured Wrapper 的纯格式效应

100 个 `no_memory` scenarios：

| 对照 | Top-1 changed | Top-3 changed | Full changed |
|---|---:|---:|---:|
| Query Only vs Empty Structured Wrapper | **43/100** | 64/100 | **100/100** |
| Query Only vs Non-empty Structured Memory（有记忆场景） | — | — | — |

这说明 **Structured Prompt 的 wrapper 本身就会改变排序**；不能把 300/300 全排序变化都归因于“Memory 生效”。本轮不据此调 Prompt。

---

## 3. 完整 Memory Adapter 与约束测试

结果：`runs/agent_ranker_phase43/memory_constraint_tests.json`

### 3.1 No-Memory Invariance

- 100 个 no_memory scenarios；
- Hybrid 的 memory adjustment 为 0；
- Query Only vs Hybrid 排序变化 **0/100**；
- 但 Query vs **Empty Structured Wrapper** 变化 **100/100**，证明格式效应独立于 Memory。

### 3.2 Withdrawal / Update

- Withdrawal 2 cases：mutated active state 为空；adapter 不再产生该偏好的 match/conflict；**pass**。
- Update 2 cases：只使用最新有效状态；旧偏好不与新偏好同时贡献；**pass**。

### 3.3 Multi-Preference

- 同时 2 positive + 1 negative：`active_preference_count=3`；
- positive match / unknown / negative conflict/unknown 分别独立计数；
- 不退回只读取第一个 Preference；**pass**。

### 3.4 Negative Soft / UNKNOWN

- Negative soft 只产生 conflict 特征，不做硬过滤；**pass**。
- UNKNOWN 不产生 match 或 conflict；不会被当作 absence；**pass**。

### 3.5 Conflict / ASK

- 16 个 Benchmark v2 conflict scenarios 全部 `requires_clarification=true`；
- 不生成正常 Top-1 Gold；
- **pass**。

---

## 4. P20 Agent E2E 实测

报告文件：`runs/agent_ranker_phase43/e2e_p20.json`

### 4.1 执行内容

- 真实 Qwen3.5-9B Base Planner 通过 `/plan` 本地模型服务；
- 真实 `ShoppingCompanionMemory.ingest_sessions` 写入 P20 历史；
- 真实 `HistoryStateReader` 读取 active preferences；
- Original BGE 检索；
- 三个 Ranker arms；
- 真实 `DecisionGate`；
- 确定性 Evidence-only final answer（不是 Qwen explain；此限制已明确）。

### 4.2 结果

| Arm | Success | Finalized | Selected in Eligible | Hard Violations | Errors | Final Answer Evidence |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 18/19 | 19/19 | 19/19 | 0 | 0 | 19/19 |
| Natural | **19/19** | 19/19 | 19/19 | 0 | 0 | 19/19 |
| Hybrid | 18/19 | 19/19 | 19/19 | 0 | 0 | 19/19 |

- 所有 finalized 选择都在合法候选池内；
- 全部 final answer 的属性来自 catalogue evidence；
- 没有 ASK 需求（19 个 recommend tasks 都预期 finalize）；
- **0051**：

```text
Existing:  B07BKPKXTT (wrong)
Natural:   B08D32JM5F (acceptable)
Hybrid:    B08D32JM5F (acceptable)
```

Natural 在 0051 E2E 中选择了 acceptable，且 Gate `ALLOW_FINALIZE`。

### 4.3 P20 结论

Natural 19/19 是最强结果，但：
- P20 是 provisional；
- 只领先 Existing 1 个 task；
- 不能仅凭此替换生产；
- 可以进入下一轮独立用户偏好评测。

---

## 5. 模型生命周期与 24GB GPU 部署

### 5.1 实测

| 项目 | 实测 |
|---|---:|
| Qwen Planner GPU 占用 | **17.55 GB** |
| ShopRanker 20-candidate batch 峰值 | **1.92 GB**（与 Planner 并发） |
| ShopRanker 300-scenario 全量峰值 | **4.58 GB** |
| 并发驻留测试 | **PASS, no OOM** |
| E2E OOM | **0** |
| E2E errors | **0** |

Qwen 9B + ShopRanker 20-candidate 并发驻留实测未 OOM，但总占用已接近 24GB；**顺序加载/卸载仍是更安全的部署路径**。

### 5.2 延迟拆分（P20 E2E）

```text
Qwen Planner p50:       3.48–3.71 s
BGE retrieval p50:      0.28–0.35 s
Ranker injection:       0.35 ms
Final answer template:  0.012 ms
E2E p50:                10.46–10.87 s
E2E p95:                13.43–13.88 s
```

- Ranker-only（ShopRanker 20 candidates）Phase 4.2 实测 p50 约 **109.98 ms**；
- Hybrid ranker p50 约 **110.09 ms**；
- E2E 耗时主要由 Qwen Planner 占用，不是 GPU Ranker 本身。
- Final answer 是确定性 evidence template，不是 Qwen 生成；因此 0.012 ms 不代表 LLM 最终答案生成延迟。

---

## 6. Counterfactual E2E

报告文件：`runs/agent_ranker_phase43/e2e_counterfactual.json`

16 个 cases，baseline + mutated，真实 Memory Writer/Reader、固定 BGE candidate pool、Rule Planner、DecisionGate。

### 6.1 Status

| State / Arm | Finalized | Awaiting Clarification | Budget Exhausted |
|---|---:|---:|---:|
| baseline / Existing | 14 | 2 | 0 |
| mutated / Existing | 14 | 2 | 0 |
| mutated / Structured | 11 | 5 | 0 |
| mutated / **Hybrid** | **0** | 0 | **16** |

- Hard constraint violations：**0**；
- conflict/ASK：4 个 records 保持 `awaiting_clarification`；
- **Hybrid 在 16/16 个 CF E2E 中 budget_exhausted，不能在 E2E 中完成 finalization**。这是本轮 Hybrid 的重大失败，不能把它作为生产候选。

### 6.2 分类

- Positive / Negative / Update / Withdrawal / Conflict / UNKNOWN 均执行了 Writer→Reader→Ranker→Gate；
- 没有独立方向 Gold；
- 冲突保持 ASK，没有强行生成正常 Top-1；
- Hybrid 的 CF 失败需要下一轮排查（不是本轮通过调权重重试）。

---

## 7. Benchmark v2 与 Ranking-Level 结果

复用 Phase 4.2 的真实 GPU 结果，不重复全部推理；输入和结果均有 hash/manifest。

| Arm | Silver Pairwise | Negative Direction | Top-1 changes vs Query |
|---|---:|---:|---:|
| Existing Scorer | 145/145* | — | — |
| Shop Query Only | 77/145 = 0.5310 | 6/17 | — |
| Shop Natural | 76/145 = 0.5241 | 8/17 | 92/300 |
| Shop Structured | 81/145 = 0.5586 | 7/17 | 127/300 |
| Fixed Hybrid | 106/145 = 0.7310 | 11/17 | 33/300 |
| DCN A | 100/145 = 0.6897 | — | — |
| DCN B | 38/145 = 0.2621 | — | — |
| DCN C | 102/145 = 0.7034 | — | — |

\* Existing 1.0 是 Silver 规则同源，tautological。

- Structured 仍没有可靠优于 Natural；
- Hybrid 在 Silver/negative 更好，但 Silver 同源，不能作为独立用户偏好证据；
- DCN B 继续失败，不进入生产候选。

---

## 8. 少量新 Agent-Native 用户选择任务

任务包：`runs/agent_ranker_phase43/blind_review_tasks.jsonl`（40 对）

- 来自 Existing / Natural / Structured / Hybrid 排序不一致；
- 优先覆盖 negative、withdrawal、UNKNOWN、update、query-memory competition；
- 展示 Query、Active Memory、A/B 商品证据；
- **隐藏所有模型身份、分数、BGE rank 和规则标签**；
- 标注选项：A / B / BOTH / INSUFFICIENT；
- 对应 CSV：`blind_review_template.csv`；指南：`blind_review_guidelines.md`。

**HUMAN_GOLD = 0；HUMAN_SINGLE_REVIEW = 0。**
本轮没有实际人工输入，因此不报告 Human Pairwise Agreement，也不把包内任务称为已标注 Gold。

---

## 9. 四层结论

### Ranking-Level

- P20 E2E：Natural 19/19，Existing 18/19，Hybrid 18/19。
- Benchmark v2 Silver：Natural 76/145；Structured 81/145；Hybrid 106/145；DCN B 38/145。
- Negative direction：Natural 8/17，Structured 7/17，Hybrid 11/17；均不足以证明可靠负向 Memory 理解。

### Agent-Level

- FINALIZE：所有 P20 E2E arms 19/19；hard violations 0；selected 均 eligible。
- ASK：P20 recommend 无需 ASK；CF conflict/ASK 保留。
- Final Answer Evidence：19/19 由 catalogue evidence 支撑；但生成器是确定性模板，不是 Qwen explain。
- Failure：Hybrid CF 16/16 budget_exhausted；Existing 0051 错误；Hybrid P20 0054 错误。

### Human Review

**Unavailable / 0 human labels.** 40 对盲评包已准备，但没有实际人工输入。

### GPU Engineering

- Qwen Planner 17.55GB；ShopRanker 4.58GB full / 1.92GB concurrent batch；无 OOM。
- E2E p50 10.46–10.87s；p95 13.43–13.88s；Planner 是主要成本。
- Ranker-only ~110ms；E2E 不是 Ranker 瓶颈。
- 顺序加载最稳；并发驻留虽 PASS，但余量小。

---

## 10. Production Candidate 决策

### Existing PreferenceAwareScorer

**保留生产默认。**
E2E 18/19，hard 0，延迟稳定；仍是确定性 baseline。

### ShopRanker Natural Memory

**值得继续 Shadow / 进入下一轮用户偏好评测；暂不进入生产试用。**

理由：

- P20 E2E 19/19，0051 正确；
- 但 Benchmark v2 Silver 76/145，negative direction 8/17；
- 没有独立 HUMAN_GOLD；
- 不能因 1 个 task 的领先替换生产。

### Fixed Hybrid

**暂不适合生产，也不建议直接进入用户试用。**

理由：

- Silver 106/145，但 Silver 与规则同源；
- Negative 11/17，仍不稳定；
- **CF E2E 16/16 budget_exhausted**；
- P20 E2E 18/19，0054 错误；
- 只能作为实验 Shadow baseline，先排查 CF 失败和补独立标签。

### DCN-V2

**不进入本轮生产候选。**
历史迁移失败已定位为分布/语义不匹配。

---

## 11. Q1–Q5 回答

### Q1. 三个方案在最终推荐中的区别？

P20 E2E：Natural 19/19、Existing 18/19、Hybrid 18/19。
Natural 在 0051 和多个 Memory 场景选择不同；Hybrid 在 0054 退化。
差异只有 1 个 task 时不能称为生产显著优势。

### Q2. Memory 更新、撤回和冲突是否正确处理？

Adapter 层约束测试全部 pass：

- No-Memory Hybrid invariance 0/100 变化；
- Withdrawal / Update / Multi-Preference / Unknown / Negative Soft 全 pass；
- 16 个 conflict 全部 ASK，未强行 Gold。

但 **Hybrid CF E2E 16/16 budget_exhausted**，说明完整 Agent 路径仍失败；不能把 Adapter 单元通过当成 E2E 通过。

### Q3. Hybrid 是否保留 ShopRanker 语义优势？

Ranking 层：Hybrid Silver 106/145，Query 77，Natural 76；Top-1 扰动 33/300，小于 Structured 127/300。
P20：Hybrid 18/19，低于 Natural 19/19；0051 正确；0054 错误。
**部分保留语义优势，但 E2E 行为不达标，不能称为稳定方案。**

### Q4. 24GB 上能否稳定运行 Planner + Ranker？

实测：

- Qwen 17.55GB + Shop 20-candidate concurrent batch 1.92GB，无 OOM；
- 全量 Shop 300 scenarios 峰值 4.58GB；
- 顺序加载/卸载是最稳路径；
- E2E p50 10.46–10.87s，p95 13.43–13.88s，瓶颈是 Planner。

**工程可行，但并发余量小，建议顺序部署。**

### Q5. 哪个方案有资格进入下一轮用户偏好评测/受控试用？

- **ShopRanker Natural**：有资格进入下一轮用户偏好评测（不是生产试用）。
- **Existing Scorer**：保留生产默认，不替换。
- **Hybrid**：暂不进入用户试用；先修复 CF E2E budget_exhausted，并获取独立标签。
- **DCN-V2**：不进入。

---

## 12. 交付文件

### 代码

```text
experiments/agent_ranker_e2e/run_e2e_p20.py
experiments/agent_ranker_e2e/run_e2e_counterfactual.py
experiments/agent_ranker_e2e/build_reproduction_diagnostics.py
experiments/agent_ranker_e2e/build_memory_constraint_tests.py
experiments/agent_ranker_e2e/build_blind_review.py
experiments/agent_ranker_e2e/build_e2e_summary.py
experiments/agent_ranker_e2e/build_runtime_manifest.py
```

### 结果

```text
runs/agent_ranker_phase43/
  e2e_p20.json
  e2e_counterfactual.json
  e2e_summary.json
  reproduction_differences.json
  memory_constraint_tests.json
  blind_review_tasks.jsonl
  blind_review_template.csv
  blind_review_guidelines.md
  gpu_runtime.json
  experiment_manifest.json
```

### 复现与约束

```text
seed: 42
Qwen planner: local Qwen3.5-9B-Base checkpoint
ShopRanker: frozen srpone/zoowork-shopranker-0.6b
DCN: reused A/B/C checkpoints, not retrained
BGE: existing index, not rebuilt
Production Scorer/Gate/Planner/P18: not modified
ShopRanker / Qwen / DCN: not retrained
Git Push: no
```

**Phase 4.3 完成：真实 Qwen Planner + Memory Writer/Reader + Ranker + Gate 的 P20 E2E 已运行；Natural 19/19、Existing/Hybrid 18/19；Hybrid CF E2E 16/16 失败；24GB 并发可运行但建议顺序部署；Final Answer 使用确定性证据模板，HUMAN_GOLD 仍为 0；生产默认保持 Existing Scorer。**
