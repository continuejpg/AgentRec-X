# P3 端到端瓶颈分析

## 0. 结论

四组（C / D0 / D1-v1 / D1-v2）在 20 个 dev 任务上的 **V2 PASS 集合完全相同**：

```text
PASS 12 / FAIL 5 / UNKNOWN 3
```

D1-v2 的 Parser 已明显改善（困难集 Exact 0.92、原 Test Exact 1.00），但 Agent 成功率
不变。LLM 的增量作用**没有停在 Parser，而是停在 M5 状态之后**：

- Proposal 可以校验并写入 M5（D1 52 个有效 operation → 38 个 M5 events）；
- 但 C vs D1 的最终商品只有 **3/20** 任务发生变化；
- 这 3 个变化没有改变任何任务的 V2 结果；
- D0 60/61 个 proposal 无效，却仍与 C 一样 12/5/3，说明当前 20 个任务的 R1/M6.1 结果
  并不依赖“LLM proposal 是否写 M5”。

---

## 1. 逐任务因果追踪

产物：

- `training/agent_sft/eval/P3_TASK_CAUSAL_TRACE.csv`
- `training/agent_sft/reports/P3_BOTTLENECK_METRICS.json`

统计：

| 项目 | 数值 |
|---|---:|
| D1 有效 proposal turns | 52 |
| D1 accepted operations | 52 |
| D1 proposal 对应 M5 events | 38 |
| 有 proposal M5 events 的任务 | 20/20 |
| D1 全部 M5 events | 213 |
| D0 proposal turns | 61 |
| D0 invalid proposals | 60 |

### 配对最终商品变化

| 对比 | 最终商品集合改变的任务 | V2 结果相同任务 |
|---|---:|---:|
| C vs D0 | 2/20 | 20/20 |
| C vs D1 | 3/20 | 20/20 |
| D0 vs D1 | 1/20 | 20/20 |
| D1 vs D1-v2 | 1/20 | 20/20 |

这意味着：

- D1 的 proposal 确实改变了 M5，并让 3 个任务的最终商品身份发生变化；
- 但 Evaluator V2 对这 3 个变化全部给出相同 PASS/FAIL/UNKNOWN；
- 从“有效状态变化”到“任务成功”的传递在 20 个 dev 任务上几乎为零。

---

## 2. 为什么 D0 大量无效 Proposal 却同分？

事实：

- D0 Base 在 61 个 proposal 轮次中 60 个 JSON 无效，被 Validator 拒绝；
- 无效 proposal 没有写入 M5，没有规则 fallback 冒充模型 proposal；
- D0 仍有 C 组规则提取器参与 R1/M6.1；
- C vs D0 只有 2/20 最终商品不同，V2 全部相同。

因此，**D0 的相同成绩不是“隐藏规则回退让模型生效”**，而是：

1. 这 20 个任务的推荐决策对 M5 状态不敏感；
2. R1 主要响应当前消息/规则提取，而不是 M5 effective state；
3. 失败任务由角色/候选/证据/生命周期决定，Parser proposal 不是决定项；
4. V2 对 final IDs 的身份变化不敏感，或这些变化落在同一 rubric 判定下。

---

## 3. LLM 增量作用停在哪

链路与证据：

| 环节 | 证据 | 结论 |
|---|---|---|
| Parser | SFT-v2 test Exact 1.00、hard .92 | 已改善 |
| Validator | D1 52/64 valid | 工作 |
| Proposal→M5 | 52 accepted → 38 events，20/20 tasks | 工作 |
| M5→Effective State | 有 events；SQLite replay 单测 | 工作 |
| Effective State→R1 | C vs D1 final IDs 仅 3/20 改变 | **弱传递/几乎无影响** |
| R1→候选 | 9 个失败任务在 C/D0/D1 完全相同 | 候选/角色是主要失败 |
| 候选→M6.1→Top-K | D1 RETRIEVE 34/RERANK 18，C 32/18 | 工具动作不是区分点 |
| Top-K→V2 | pass 集合完全相同 | Benchmark 不区分这些变化 |

**LLM 增量作用停在 M5 effective state 到 R1/M6.1 的使用阶段。** 不是 Parser 不够好。

---

## 4. 失败任务（四组完全相同）

失败/UNKNOWN 集合：

```text
dev_05_budget_conflict              UNKNOWN
dev_06_cookware_material            FAIL (lifecycle)
dev_07_cookware_lightweight_pref    FAIL (lifecycle)
dev_12_goal_change_sleeping_to_lantern UNKNOWN (role/candidate)
dev_13_goal_change_tablets_to_filter FAIL (role)
dev_14_gravity_filter_pref          FAIL (lifecycle)
dev_17_filter_budget_pref           UNKNOWN
dev_18_filter_color_correction      FAIL (role)
```

这些任务在 C、D0、D1、D1-v2 中同为 FAIL/UNKNOWN，且失败 component 也一致：

- lifecycle 17/20 PASS；
- product role 17/20 PASS；
- retrieval 17/20 ADEQUATE；
- hard constraint 18/20 PASS。

所以不是 SFT 或 D1 的 proposal 引入的问题；问题在状态语义/角色证据/候选覆盖。

---

## 5. Benchmark 敏感性

当前 V2：

- C/D0/D1/D1-v2 全部 12/5/3；
- 2-3 个任务最终商品集合变化，V2 判定不变；
- 失败任务集合完全一致；
- 这说明当前 20 个 dev 任务对 Parser 改进不敏感，不能用于证明 Agent 增益。

但这不等于 V2 本身错误：它反映的是当前任务/状态链路下角色、生命周期、证据的硬约束；
更准确地说，**任务集和状态消费方式没有给 Parser 增益留下可观测空间**。

---

## 6. 结论

1. Parser 提升是真实的；
2. Proposal 到 M5 也是真实的；
3. 但 M5 effective state 没有充分影响 R1/M6.1 的最终身份；
4. 当前失败主要是 M5 lifecycle、product role、候选召回/证据；
5. 继续训练 Parser SFT-v3 不会解决此问题；
6. 在修好 effective state→R1/M6.1 之前，Agent RL 缺乏可信 reward 和动作因果。
