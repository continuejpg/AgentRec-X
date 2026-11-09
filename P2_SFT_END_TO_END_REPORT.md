# P2-SFT 泛化与端到端验证报告

## 0. 结论摘要

| 实验 | 状态 | 结论 |
|---|---|---|
| 1. 分类别错误分析 | DONE | SFT 的 8 个有 support 操作类型接近满分；主要弱点是 clarify↔noop；Macro 低主要由零-support 类别和 clarify/noop 少数错误造成 |
| 2. 独立困难测试 | DONE（exploratory） | SFT Exact 从模板 test 的 0.945 降到 0.66；仍显著高于 Base 的 0.08，也高于困难集 Rule 的 0.08 |
| 3. Agent C/D0/D1 端到端 | **NOT COMPLETED** | proposal→M5/R1/M6.1 执行桥未实现；没有伪造 D0/D1 指标 |

**重要限制**：困难集 100 条为新建、待人工审核的 exploratory 数据；20 条审核记录仍为
pending，不能当作正式人工 Gold。未根据困难集结果调整模型或 Prompt。

---

## 1. 实验一：分类别错误分析

详见：

- [P2_SFT_ERROR_ANALYSIS.md](P2_SFT_ERROR_ANALYSIS.md)
- [P2_SFT_TEST_OPERATION_METRICS.csv](training/agent_sft/reports/P2_SFT_TEST_OPERATION_METRICS.csv)
- [P2_SFT_TEST_ERROR_SUMMARY.json](training/agent_sft/reports/P2_SFT_TEST_ERROR_SUMMARY.json)

核心结果（200 条原 Parser Test，复用已有预测）：

| provider | valid JSON | Exact | Macro all-12 | Macro supported-9 |
|---|---:|---:|---:|---:|
| Rule | 200 | 0.820 | 0.750 | 1.000 |
| Base | 66 | 0.095 | 0.254 | 0.339 |
| SFT | 194 | 0.945 | 0.703 | **0.938** |

SFT 主错误：

```text
clarify -> noop   5
其他操作类型      0
```

SFT 的 clarify recall 为 0.389，noop precision 为 0.783；其余 8 个有 support 操作
F1 均为 1.0。

Exact 94.5% 与 Macro 70.3% 的差异来自：

- Exact 是 200 个样本级整体匹配；
- Macro 是 12 个操作等权；其中 3 个操作在测试集没有 Gold support，F1=0；
- clarify 的 5 个语义错误被 macro 放大，而 Exact 只计 5/200。

原始 macro 0.703 保留；新增 supported macro 0.938 不覆盖原指标。

---

## 2. 实验二：独立困难测试

数据集：

```text
version: p2sft.hard_test.v1
samples: 100
sha256: 9162c867224c2efb336a9fb75e85f8aad15eb1f2f33c1700235c3a48f61dbcad
human_review_status: pending
human_reviewed_count: 0
```

- 由独立手工语义配置生成；
- 不复制训练/验证/测试模板；
- 不使用 frozen tasks；
- 不使用 C 组输出作为 Gold；
- Gold 在正式评测前冻结；
- 100 条中 20 条审核清单已生成，尚未完成人工审核；
- 因此以下指标是 **exploratory**，不是正式人工 Gold 指标。

困难覆盖：

- 多条件同时更新；
- 预算反复修改；
- 复杂偏好撤回；
- 跨轮指代；
- 隐含但未确认偏好；
- 否定/矛盾；
- 商品目标与配件角色变化；
- CLARIFY / NOOP / ABSTAIN；
- 价格+材料+重量组合。

### 实测结果

| metric | Rule | Base Qwen3.5-9B | SFT Qwen3.5-9B |
|---|---:|---:|---:|
| JSON Valid Rate | 1.00 | 0.26 | **0.93** |
| Operation Exact Match | 0.08 | 0.08 | **0.66** |
| Macro-F1 (all) | 0.192 | 0.239 | **0.635** |
| Macro-F1 (supported) | 0.256 | 0.318 | **0.847** |
| Slot Accuracy | 0.375 | 0.243 | **0.535** |
| Retraction Accuracy | 0.133 | 0.800 | **1.000** |
| Unsupported Proposal Rate | 0.00 | 0.74 | **0.07** |
| Failure/Abstention Rate | 0.00 | 0.74 | **0.07** |
| p50 latency ms | 0.026 | 2211.1 | 3043.5 |
| p95 latency ms | 0.030 | 3675.1 | 4410.3 |

### 与模板 Parser Test 的变化

| metric | 模板 test SFT | hard test SFT | 变化 |
|---|---:|---:|---:|
| JSON Valid | 0.97 | 0.93 | -0.04 |
| Exact Match | 0.945 | 0.66 | **-0.285** |
| Macro-F1 all | 0.703 | 0.635 | -0.068 |
| Macro-F1 supported | 0.938 | 0.847 | **-0.091** |
| Retraction Accuracy | 1.00 | 1.00 | 0 |
| Unsupported Rate | 0.03 | 0.07 | +0.04 |

困难集主要 SFT 错误：

- `update_constraint -> add_preference` 21 次；
- 多条件组合下 update_constraint 被漏掉或降级成 preference；
- `resolve_reference` recall 0.4；
- clarify/noop 在困难否定/隐含场景仍有边界错误；
- 其余 `add_preference`、`retract_preference`、`retract_constraint`、`clarify`、
  `abstain` 表现较好。

完整结果：

- [P2_SFT_HARD_TEST_RESULTS.csv](training/agent_sft/eval/P2_SFT_HARD_TEST_RESULTS.csv)
- [P2_SFT_HARD_OPERATION_METRICS.csv](training/agent_sft/reports/P2_SFT_HARD_OPERATION_METRICS.csv)
- 预测：
  - `training/agent_sft/eval/hard_base/predictions.jsonl`
  - `training/agent_sft/eval/hard_sft/predictions.jsonl`
- 人工审核清单：`training/agent_sft/hard_test/hard_test_manual_review_20.csv`

**不根据困难集调整 Prompt 或模型；本结果仅 exploratory。**

---

## 3. 实验三：Agent 端到端对照

### 状态：未完成

原要求：

```text
C: Rule Agent
D0: Base Qwen3.5-9B Agent
D1: SFT Qwen3.5-9B Agent
```

当前现实：

- C 组已有历史执行结果（P1C/DEV），可作为规则参考；
- D 组适配器目前只把 LLM 的 `rewritten_message` 送入 C 管线；
- `preferences` proposal 仅写入 `llm_proposals` 日志，**没有进入 M5 events**；
- 因此没有真正的 `Proposal → Validator → M5 → R1/M6.1` 执行桥；
- 本阶段未实现该桥，也没有伪造 D0/D1 结果。

C 参考（P1C 当前 C 配置）：

```text
legacy deterministic: 8/20
Evaluator V2: PASS 12 / FAIL 5 / UNKNOWN 3
```

逐任务状态在：

- [P2_SFT_AGENT_COMPARISON.csv](training/agent_sft/eval/P2_SFT_AGENT_COMPARISON.csv)
  - C 行来自 P1C 已有运行；
  - D0/D1 明确标记 `not_run`，原因为 proposal→M5 执行桥未实现。

### 为什么不能把 Parser 结果冒充 Agent 结果

Parser 只验证“自然语言 → operation JSON”。Agent 效果还需要：

1. proposal validator 拒绝无效操作；
2. operation 转化为 M5 events；
3. M5 lifecycle/replay 更新有效状态；
4. R1/M6.1 使用更新后的状态；
5. 最终角色/硬约束/Top-K 发生可观测变化。

当前只完成了第 0/1 层（模型 + schema），没有完成第 2-5 层的执行接入。因此不能回答
“SFT 是否改善 Agent 任务成功率”；只能说 parser 明显优于 Base。

---

## 4. 最终回答

1. **SFT 哪些操作学得好，哪些不足？**
   - 好：set_goal、update_goal、add_preference、update_constraint、retract_constraint、
     resolve_reference、abstain（模板测试 F1 均 1.0）；
   - 不足：clarify↔noop 区分（5 个错误）；困难集多条件 update_constraint 漏判、
     resolve_reference recall 0.4。

2. **独立困难表达是否仍然有效？**
   - 是，但明显下降：Exact 0.945→0.66，supported Macro 0.938→0.847；
   - 仍远高于 Base（Exact 0.08，supported Macro 0.318）和困难集 Rule（Exact 0.08）；
   - 困难集尚待人工审核，不能当作正式泛化结论。

3. **SFT 是否改善真实 Agent 的 M5 状态和任务成功率？**
   - **不能判断**；D0/D1 执行桥未实现，未运行 Agent 对照；
   - 不把 Parser 指标冒充 Agent 指标。

4. **当前主要问题是训练数据覆盖不足、模型理解不足还是执行链路不足？**
   - Parser 层：模型已明显改善，剩余是 clarify/noop、多条件组合、跨轮指代等
     **数据覆盖和困难负例不足**；
   - Agent 层：**执行链路不足是首要问题**，proposal 未进入 M5；
   - 不是“换更大模型”能直接解决。

5. **是否有必要进入下一轮数据扩充或 Agent RL？**
   - 在完成 proposal→M5 执行桥之前，**不应进入 Agent RL**；
   - 下一轮应优先补困难 clarify/noop、多条件更新、跨轮指代、撤回目标数据，并做
     Agent D0/D1 真实执行；当前证据不足以支持 RL。

---

## 5. 未完成 / 约束

- 未运行 frozen Agent Benchmark；
- 未修改 Serving V1、SASRec、Two-Tower、BM25、ItemCF；
- 未训练新模型、未扩训练集；
- 未开始 Agent RL；
- 未 commit/push/deploy；
- 原始 Parser Test、Gold、数据 hash 未修改；
- 困难集共 100 条，仅准备 20 条人工审核清单，正式人工审核未完成；
- D0/D1 Agent 结果不存在。
